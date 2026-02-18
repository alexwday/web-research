/**
 * Quarterly Disclosure Monitor — visual UI state machine.
 *
 * Manages bank icon bar, per-bank activity tree, and inline PDF/XLSX preview.
 * Each bank scan runs independently with its own poll timer.
 */

/* ── Global state ─────────────────────────────────────────────── */

const QM = {
    banks: [],
    docTargets: [],
    bankScans: {},       // bankCode -> { status, jobId, lastLogIndex, tree }
    activeBankCode: null,
    pollTimers: {},      // bankCode -> intervalId
    initialized: false,
    configMode: false,   // true = preview panel shows config form
    configBankCode: null, // bank being edited (null = new bank)
};

/* ── Init ─────────────────────────────────────────────────────── */

async function initQuarterly() {
    if (QM.initialized) return;
    try {
        const [banksRes, targetsRes] = await Promise.all([
            fetch('/api/config/banks').then(r => r.json()),
            fetch('/api/config/doc-targets').then(r => r.json()),
        ]);
        QM.banks = banksRes;
        QM.docTargets = targetsRes;
        QM.initialized = true;
        _buildBankBar();
    } catch (err) {
        console.error('initQuarterly failed:', err);
    }
}

/* ── Open / Close ─────────────────────────────────────────────── */

function openQuarterly() {
    initQuarterly().then(() => {
        document.getElementById('quarterly-modal').hidden = false;
    });
}

function closeQuarterly() {
    // Stop all poll timers
    Object.keys(QM.pollTimers).forEach(code => {
        clearInterval(QM.pollTimers[code]);
        delete QM.pollTimers[code];
    });
    document.getElementById('quarterly-modal').hidden = true;
}

/* ── Bank bar ─────────────────────────────────────────────────── */

function _buildBankBar() {
    const bar = document.getElementById('qm-bank-bar');
    if (!bar) return;
    bar.innerHTML = '';

    QM.banks.forEach(bank => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'qm-bank-btn qm-bank--idle';
        btn.dataset.bankCode = bank.code;

        // Restore scan status if bank was already scanned
        if (QM.bankScans[bank.code]) {
            const s = QM.bankScans[bank.code].status;
            btn.classList.remove('qm-bank--idle');
            btn.classList.add(`qm-bank--${s}`);
        }

        const icon = document.createElement('span');
        icon.className = 'qm-bank-icon';
        icon.textContent = bank.code.substring(0, 3).toUpperCase();

        const label = document.createElement('span');
        label.className = 'qm-bank-label';
        label.textContent = bank.name;

        // Gear icon overlay
        const gear = document.createElement('span');
        gear.className = 'qm-bank-gear';
        gear.innerHTML = '&#x2699;';
        gear.title = 'Configure ' + bank.name;
        gear.addEventListener('click', (e) => {
            e.stopPropagation();
            _openConfigForm(bank.code);
        });

        btn.appendChild(icon);
        btn.appendChild(label);
        btn.appendChild(gear);
        btn.addEventListener('click', () => _onBankClick(bank.code));
        bar.appendChild(btn);
    });

    // "+" button to add a new bank
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'qm-bank-btn qm-bank--add';
    addBtn.title = 'Add new bank';
    const addIcon = document.createElement('span');
    addIcon.className = 'qm-bank-icon';
    addIcon.textContent = '+';
    const addLabel = document.createElement('span');
    addLabel.className = 'qm-bank-label';
    addLabel.textContent = 'Add Bank';
    addBtn.appendChild(addIcon);
    addBtn.appendChild(addLabel);
    addBtn.addEventListener('click', () => _openConfigForm(null));
    bar.appendChild(addBtn);
}

function _onBankClick(bankCode) {
    QM.activeBankCode = bankCode;
    _highlightActiveBank();

    if (!QM.bankScans[bankCode]) {
        startBankScan(bankCode);
    } else {
        renderTree(bankCode);
    }
}

function _highlightActiveBank() {
    document.querySelectorAll('.qm-bank-btn').forEach(btn => {
        btn.classList.toggle('qm-bank--active', btn.dataset.bankCode === QM.activeBankCode);
    });
}

function _updateBankBtnStatus(bankCode, status) {
    const btn = document.querySelector(`.qm-bank-btn[data-bank-code="${bankCode}"]`);
    if (!btn) return;
    btn.classList.remove('qm-bank--idle', 'qm-bank--scanning', 'qm-bank--complete', 'qm-bank--partial');
    btn.classList.add(`qm-bank--${status}`);
}

/* ── Start bank scan ──────────────────────────────────────────── */

async function startBankScan(bankCode) {
    const quarter = document.getElementById('qm-quarter')?.value || 'Q4';
    const year = document.getElementById('qm-year')?.value || '2025';

    // Init tree skeleton from doc targets
    const tree = {};
    QM.docTargets.forEach(dt => {
        const formats = {};
        dt.required_formats.forEach(fmt => {
            formats[fmt] = {
                status: 'pending',
                phases: [],
            };
        });
        tree[dt.doc_type] = {
            label: dt.label,
            domainQueries: [],
            candidateCount: 0,
            formats: formats,
        };
    });

    QM.bankScans[bankCode] = {
        status: 'scanning',
        jobId: null,
        lastLogIndex: 0,
        tree: tree,
        _queryBuffer: {},  // doc_type -> [{query, resultCount, urls}]
    };

    _updateBankBtnStatus(bankCode, 'scanning');
    renderTree(bankCode);

    try {
        const res = await fetch('/api/use-cases/quarterly-docs/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bank: bankCode, quarter, year }),
        });
        const data = await res.json();
        QM.bankScans[bankCode].jobId = data.job_id;
        _startPolling(bankCode, data.job_id);
    } catch (err) {
        console.error('startBankScan failed:', err);
        QM.bankScans[bankCode].status = 'partial';
        _updateBankBtnStatus(bankCode, 'partial');
    }
}

/* ── Polling ──────────────────────────────────────────────────── */

function _startPolling(bankCode, jobId) {
    if (QM.pollTimers[bankCode]) clearInterval(QM.pollTimers[bankCode]);
    _pollOnce(bankCode, jobId);
    QM.pollTimers[bankCode] = setInterval(() => _pollOnce(bankCode, jobId), 1200);
}

async function _pollOnce(bankCode, jobId) {
    try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) return;
        const job = await res.json();
        processNewLogs(bankCode, job);

        if (bankCode === QM.activeBankCode) {
            renderTree(bankCode);
        }

        if (job.status === 'completed' || job.status === 'failed') {
            clearInterval(QM.pollTimers[bankCode]);
            delete QM.pollTimers[bankCode];
        }
    } catch (err) {
        console.error('poll error:', err);
    }
}

/* ── Log processing (phase routing) ───────────────────────────── */

function processNewLogs(bankCode, job) {
    const scan = QM.bankScans[bankCode];
    if (!scan) return;

    const logs = job.logs || [];
    for (let i = scan.lastLogIndex; i < logs.length; i++) {
        const entry = logs[i];
        const d = entry.data || {};
        const phase = d.phase;
        if (!phase) continue;

        // Only process events for this bank
        if (d.bank_code && d.bank_code !== bankCode) continue;

        const tree = scan.tree;
        const docNode = d.doc_type ? tree[d.doc_type] : null;

        switch (phase) {
            case 'bank_start':
                scan.status = 'scanning';
                _updateBankBtnStatus(bankCode, 'scanning');
                break;

            case 'domain_search_start':
                // Mark doc_type active
                break;

            case 'subquery':
                if (docNode) {
                    if (!scan._queryBuffer[d.doc_type]) scan._queryBuffer[d.doc_type] = [];
                    scan._queryBuffer[d.doc_type].push({
                        query: d.query || entry.message,
                        resultCount: 0,
                        urls: [],
                    });
                }
                break;

            case 'subquery_results':
                if (docNode && scan._queryBuffer[d.doc_type]) {
                    const queries = scan._queryBuffer[d.doc_type];
                    if (queries.length > 0) {
                        queries[queries.length - 1].resultCount = d.result_count || 0;
                    }
                }
                break;

            case 'domain_ranked':
                if (docNode) {
                    docNode.candidateCount = d.candidate_count || 0;
                }
                break;

            case 'format_search_start':
                if (docNode && d.format && docNode.formats[d.format]) {
                    const fmtNode = docNode.formats[d.format];
                    fmtNode.status = 'searching';
                    // Attach buffered queries as first phase
                    const buffered = scan._queryBuffer[d.doc_type] || [];
                    if (buffered.length > 0) {
                        fmtNode.phases.push({
                            type: 'domain_search',
                            label: 'Domain Search',
                            queries: buffered.slice(),
                        });
                    }
                }
                break;

            case 'verify_candidate':
            case 'download':
                if (docNode && d.format && docNode.formats[d.format]) {
                    const fmtNode = docNode.formats[d.format];
                    const url = d.url || '';
                    const existing = _findUrlEntry(fmtNode, url);
                    if (existing) {
                        existing.status = phase === 'download' ? 'downloading' : 'checking';
                    } else {
                        _addUrlEntry(fmtNode, {
                            url: url,
                            title: d.title || '',
                            status: phase === 'download' ? 'downloading' : 'checking',
                            localPath: null,
                        });
                    }
                }
                break;

            case 'download_failed':
                if (docNode && d.format && docNode.formats[d.format]) {
                    const fmtNode = docNode.formats[d.format];
                    const existing = _findUrlEntry(fmtNode, d.url);
                    if (existing) existing.status = 'download_failed';
                }
                break;

            case 'llm_verify':
            case 'extract_content':
                if (docNode && d.format && docNode.formats[d.format]) {
                    const existing = _findUrlEntry(docNode.formats[d.format], d.url);
                    if (existing) existing.status = 'verifying';
                }
                break;

            case 'verified':
            case 'extraction_failed':
            case 'llm_unavailable':
                if (docNode && d.format && docNode.formats[d.format]) {
                    const existing = _findUrlEntry(docNode.formats[d.format], d.url);
                    if (existing) {
                        existing.status = 'verified';
                        existing.localPath = d.local_path || null;
                    }
                }
                break;

            case 'not_verified':
                if (docNode && d.format && docNode.formats[d.format]) {
                    const existing = _findUrlEntry(docNode.formats[d.format], d.url);
                    if (existing) existing.status = 'rejected';
                }
                break;

            case 'web_search_start':
                if (docNode && d.format && docNode.formats[d.format]) {
                    docNode.formats[d.format].phases.push({
                        type: 'web_search',
                        label: 'Web Search',
                        queries: [],
                    });
                }
                break;

            case 'cross_pollination':
                if (docNode && d.format && docNode.formats[d.format]) {
                    docNode.formats[d.format].phases.push({
                        type: 'cross_pollination',
                        label: 'Cross-Pollination',
                        queries: [],
                    });
                }
                break;

            case 'ir_scrape_start':
                if (docNode && d.format && docNode.formats[d.format]) {
                    docNode.formats[d.format].phases.push({
                        type: 'ir_scrape',
                        label: 'IR Page Scrape',
                        pages: [],
                    });
                }
                break;

            case 'ir_scrape_links':
                if (docNode && d.format && docNode.formats[d.format]) {
                    const fmtNode = docNode.formats[d.format];
                    const irPhase = _findLastPhase(fmtNode, 'ir_scrape');
                    if (irPhase) {
                        irPhase.pages.push({
                            pageUrl: d.page_url || '',
                            linkCount: d.link_count || 0,
                            links: (d.links || []).map(l => ({
                                url: l.url || l,
                                status: 'pending',
                            })),
                        });
                    }
                }
                break;

            case 'url_inference':
                if (docNode && d.format && docNode.formats[d.format]) {
                    docNode.formats[d.format].phases.push({
                        type: 'url_inference',
                        label: 'URL Inference',
                        urls: (d.urls || []).map(u => ({
                            url: typeof u === 'string' ? u : u.url,
                            status: 'pending',
                        })),
                    });
                }
                break;

            case 'format_complete':
                if (docNode && d.format && docNode.formats[d.format]) {
                    docNode.formats[d.format].status = d.found ? 'found' : 'missing';
                }
                break;

            case 'batch_search_complete':
                // All domain searches done for this bank
                break;

            case 'verify_round_start':
                // Multiple (doc_type, format) pairs starting verification in parallel
                if (d.pairs) {
                    d.pairs.forEach(p => {
                        const node = tree[p.doc_type];
                        if (node && node.formats[p.format]) {
                            node.formats[p.format].status = 'searching';
                        }
                    });
                }
                break;

            case 'cross_route':
                // A document was routed from one doc_type to another
                if (d.target_doc_type && tree[d.target_doc_type]) {
                    const targetNode = tree[d.target_doc_type];
                    if (d.format && targetNode.formats[d.format]) {
                        targetNode.formats[d.format].phases.push({
                            type: 'cross_route',
                            label: 'Cross-Routed from ' + (d.source_doc_type || 'other'),
                            queries: [{ query: '', resultCount: 0, urls: [{ url: d.url || '', title: '', status: 'checking', localPath: null }] }],
                        });
                    }
                }
                break;

            case 'cross_route_verified':
                // Cross-routed doc was verified for its correct type
                if (docNode && d.format && docNode.formats[d.format]) {
                    const existing = _findUrlEntry(docNode.formats[d.format], d.url);
                    if (existing) {
                        existing.status = 'verified';
                        existing.localPath = d.local_path || null;
                    }
                }
                break;

            case 'bank_complete':
                scan.status = d.all_found ? 'complete' : 'partial';
                _updateBankBtnStatus(bankCode, scan.status);
                break;

            case 'completed':
            case 'failed':
                if (scan.status === 'scanning') {
                    scan.status = phase === 'completed' ? 'complete' : 'partial';
                    _updateBankBtnStatus(bankCode, scan.status);
                }
                break;
        }
    }
    scan.lastLogIndex = logs.length;
}

/* ── Tree helpers ─────────────────────────────────────────────── */

function _findUrlEntry(fmtNode, url) {
    if (!url) return null;
    for (const phase of fmtNode.phases) {
        if (phase.queries) {
            for (const q of phase.queries) {
                for (const u of (q.urls || [])) {
                    if (u.url === url) return u;
                }
            }
        }
        if (phase.pages) {
            for (const pg of phase.pages) {
                for (const l of (pg.links || [])) {
                    if (l.url === url) return l;
                }
            }
        }
        if (phase.urls) {
            for (const u of phase.urls) {
                if (u.url === url) return u;
            }
        }
    }
    return null;
}

function _addUrlEntry(fmtNode, entry) {
    // Add to the last phase's query list, or create a default phase
    if (fmtNode.phases.length === 0) {
        fmtNode.phases.push({ type: 'domain_search', label: 'Domain Search', queries: [{ query: '', resultCount: 0, urls: [] }] });
    }
    const lastPhase = fmtNode.phases[fmtNode.phases.length - 1];
    if (lastPhase.queries) {
        if (lastPhase.queries.length === 0) {
            lastPhase.queries.push({ query: '', resultCount: 0, urls: [] });
        }
        lastPhase.queries[lastPhase.queries.length - 1].urls.push(entry);
    } else if (lastPhase.pages) {
        if (lastPhase.pages.length === 0) lastPhase.pages.push({ pageUrl: '', linkCount: 0, links: [] });
        lastPhase.pages[lastPhase.pages.length - 1].links.push(entry);
    } else if (lastPhase.urls) {
        lastPhase.urls.push(entry);
    }
}

function _findLastPhase(fmtNode, type) {
    for (let i = fmtNode.phases.length - 1; i >= 0; i--) {
        if (fmtNode.phases[i].type === type) return fmtNode.phases[i];
    }
    return null;
}

/* ── Render tree ──────────────────────────────────────────────── */

function renderTree(bankCode) {
    const container = document.getElementById('qm-tree');
    if (!container) return;

    const scan = QM.bankScans[bankCode];
    if (!scan) {
        container.innerHTML = '<p class="qm-empty">Click a bank to start scanning.</p>';
        return;
    }

    const tree = scan.tree;
    let html = '';

    for (const [docType, node] of Object.entries(tree)) {
        html += `<details class="qm-doc-section" open>`;
        html += `<summary>${_esc(node.label)}`;
        if (node.candidateCount > 0) {
            html += ` <span class="qm-candidate-count">${node.candidateCount} candidates</span>`;
        }
        html += `</summary>`;
        html += `<div class="qm-doc-section-body">`;

        for (const [fmt, fmtNode] of Object.entries(node.formats)) {
            html += `<div style="margin-bottom:8px">`;
            html += `<span class="qm-format-badge qm-format-badge-${fmt}">${fmt}</span> `;
            html += _renderFmtStatus(fmtNode.status);

            for (const phase of fmtNode.phases) {
                html += `<div class="qm-phase-section">`;
                html += `<div class="qm-phase-label">${_esc(phase.label)}</div>`;

                if (phase.queries) {
                    for (const q of phase.queries) {
                        if (q.query) {
                            html += `<div class="qm-url-entry"><span class="qm-arrow">&#x1F50D;</span> <span class="qm-url-text">${_esc(q.query)}</span>`;
                            if (q.resultCount > 0) html += ` <span class="qm-candidate-count">${q.resultCount} results</span>`;
                            html += `</div>`;
                        }
                        for (const u of (q.urls || [])) {
                            html += _renderUrlEntry(u);
                        }
                    }
                }

                if (phase.pages) {
                    for (const pg of phase.pages) {
                        html += `<div class="qm-url-entry"><span class="qm-arrow">&#x1F4C4;</span> <span class="qm-url-text"><a href="${_escAttr(pg.pageUrl)}" target="_blank">${_esc(_truncUrl(pg.pageUrl))}</a></span>`;
                        if (pg.linkCount > 0) html += ` <span class="qm-candidate-count">${pg.linkCount} links</span>`;
                        html += `</div>`;
                        for (const l of (pg.links || [])) {
                            html += _renderUrlEntry(l);
                        }
                    }
                }

                if (phase.urls) {
                    for (const u of phase.urls) {
                        html += _renderUrlEntry(u);
                    }
                }

                html += `</div>`;
            }

            html += `</div>`;
        }

        html += `</div></details>`;
    }

    if (!html) {
        html = '<p class="qm-empty">Waiting for scan data...</p>';
    }

    container.innerHTML = html;
}

function _renderFmtStatus(status) {
    switch (status) {
        case 'searching': return '<span class="qm-spinner"></span>';
        case 'found': return '<span class="qm-icon-ok">&#x2713;</span>';
        case 'missing': return '<span class="qm-icon-fail">&#x2717;</span>';
        default: return '';
    }
}

function _renderUrlEntry(u) {
    let html = '<div class="qm-url-entry">';
    html += '<span class="qm-arrow">&rarr;</span>';
    html += `<span class="qm-url-text"><a href="${_escAttr(u.url)}" target="_blank" title="${_escAttr(u.url)}">${_esc(u.title || _truncUrl(u.url))}</a></span>`;
    html += `<span class="qm-url-status">${_renderStatusIcon(u.status)}</span>`;

    if (u.status === 'verified' && u.localPath) {
        const ext = u.localPath.split('.').pop().toLowerCase();
        if (ext === 'pdf') {
            html += `<button class="qm-url-preview-btn" onclick="previewPdf('${_escAttr(u.localPath)}')">&#x1F441;</button>`;
        } else if (ext === 'xlsx' || ext === 'xls') {
            html += `<button class="qm-url-preview-btn" onclick="previewXlsx('${_escAttr(u.localPath)}')">&#x1F441;</button>`;
        }
    }

    html += '</div>';
    return html;
}

function _renderStatusIcon(status) {
    switch (status) {
        case 'checking':
        case 'downloading':
        case 'verifying':
            return '<span class="qm-spinner"></span>';
        case 'verified':
            return '<span class="qm-icon-ok">&#x2713;</span>';
        case 'rejected':
        case 'download_failed':
            return '<span class="qm-icon-fail">&#x2717;</span>';
        default:
            return '';
    }
}

/* ── Preview: PDF ─────────────────────────────────────────────── */

function previewPdf(path) {
    const container = document.getElementById('qm-preview-content');
    if (!container) return;
    container.innerHTML = `<iframe src="/api/files/inline?path=${encodeURIComponent(path)}" title="PDF Preview"></iframe>`;
}

/* ── Preview: XLSX ────────────────────────────────────────────── */

async function previewXlsx(path, sheet) {
    const container = document.getElementById('qm-preview-content');
    if (!container) return;
    container.innerHTML = '<p class="qm-empty"><span class="qm-spinner"></span> Loading spreadsheet...</p>';

    try {
        let url = `/api/files/xlsx-preview?path=${encodeURIComponent(path)}&max_rows=20`;
        if (sheet) url += `&sheet=${encodeURIComponent(sheet)}`;

        const res = await fetch(url);
        if (!res.ok) throw new Error('Preview failed');
        const data = await res.json();

        let html = '';
        if (data.sheet_names && data.sheet_names.length > 1) {
            html += '<div class="xlsx-sheet-tabs">';
            data.sheet_names.forEach(name => {
                const cls = name === data.active_sheet ? 'xlsx-sheet-tab active' : 'xlsx-sheet-tab';
                html += `<button class="${cls}" onclick="previewXlsx('${_escAttr(path)}', '${_escAttr(name)}')">${_esc(name)}</button>`;
            });
            html += '</div>';
        }
        html += data.html;
        container.innerHTML = html;
    } catch (err) {
        container.innerHTML = `<p class="qm-empty">Failed to load preview: ${_esc(String(err))}</p>`;
    }
}

/* ── Config form ──────────────────────────────────────────────── */

function _openConfigForm(bankCode) {
    QM.configMode = true;
    QM.configBankCode = bankCode;
    _renderConfigForm();
}

function _closeConfigForm() {
    QM.configMode = false;
    QM.configBankCode = null;
    const container = document.getElementById('qm-preview-content');
    if (container) container.innerHTML = '<p class="qm-empty">Select a file to preview.</p>';
}

function _renderConfigForm() {
    const container = document.getElementById('qm-preview-content');
    if (!container) return;

    const isNew = QM.configBankCode === null;
    const bank = isNew ? null : QM.banks.find(b => b.code === QM.configBankCode);

    const code = bank ? bank.code : '';
    const name = bank ? bank.name : '';
    const aliases = bank ? (bank.aliases || []).join(', ') : '';
    const domains = bank ? (bank.primary_domains || []).join(', ') : '';
    const irPages = bank ? (bank.ir_pages || []).join('\n') : '';
    const docNaming = bank ? (bank.doc_naming || {}) : {};

    let html = `<div class="qm-config-form">`;
    html += `<h3>${isNew ? 'Add New Bank' : 'Configure ' + _esc(name)}</h3>`;

    html += `<div class="qm-config-field">`;
    html += `<label>Code</label>`;
    html += `<input type="text" id="qm-cfg-code" value="${_escAttr(code)}" ${isNew ? '' : 'readonly'} placeholder="e.g. rbc">`;
    html += `</div>`;

    html += `<div class="qm-config-field">`;
    html += `<label>Name</label>`;
    html += `<input type="text" id="qm-cfg-name" value="${_escAttr(name)}" placeholder="e.g. Royal Bank of Canada">`;
    html += `</div>`;

    html += `<div class="qm-config-field">`;
    html += `<label>Aliases (comma-separated)</label>`;
    html += `<input type="text" id="qm-cfg-aliases" value="${_escAttr(aliases)}" placeholder="e.g. RBC, Royal Bank">`;
    html += `</div>`;

    html += `<div class="qm-config-field">`;
    html += `<label>Primary Domains (comma-separated)</label>`;
    html += `<input type="text" id="qm-cfg-domains" value="${_escAttr(domains)}" placeholder="e.g. rbc.com">`;
    html += `</div>`;

    html += `<div class="qm-config-field">`;
    html += `<label>IR Pages (one per line, supports {year} placeholder)</label>`;
    html += `<textarea id="qm-cfg-ir-pages" rows="3" placeholder="e.g. https://example.com/investor-relations/quarterly-results-{year}">${_esc(irPages)}</textarea>`;
    html += `</div>`;

    // Doc Naming section
    html += `<div class="qm-config-section">`;
    html += `<h4>Document Naming Overrides</h4>`;

    QM.docTargets.forEach(dt => {
        const entry = docNaming[dt.doc_type] || {};
        const docAliases = (entry.document_aliases || []).join(', ');
        const urlPatterns = (entry.url_patterns || []).join(', ');

        html += `<details class="qm-config-doc-type">`;
        html += `<summary>${_esc(dt.label)} <span class="qm-config-doc-code">${_esc(dt.doc_type)}</span></summary>`;
        html += `<div class="qm-config-field">`;
        html += `<label>Document Aliases (comma-separated)</label>`;
        html += `<input type="text" data-doc-type="${_escAttr(dt.doc_type)}" data-field="document_aliases" value="${_escAttr(docAliases)}" placeholder="e.g. Supplemental Regulatory Disclosure">`;
        html += `</div>`;
        html += `<div class="qm-config-field">`;
        html += `<label>URL Patterns (comma-separated)</label>`;
        html += `<input type="text" data-doc-type="${_escAttr(dt.doc_type)}" data-field="url_patterns" value="${_escAttr(urlPatterns)}" placeholder="e.g. supp-regulatory, regulatory-disclosure">`;
        html += `</div>`;
        html += `</details>`;
    });

    html += `</div>`;

    // Buttons
    html += `<div class="qm-config-actions">`;
    html += `<button class="qm-config-save" onclick="_saveConfigForm()">Save</button>`;
    html += `<button class="qm-config-cancel" onclick="_closeConfigForm()">Cancel</button>`;
    html += `</div>`;

    html += `</div>`;
    container.innerHTML = html;
}

function _splitComma(str) {
    return String(str || '').split(',').map(s => s.trim()).filter(Boolean);
}

async function _saveConfigForm() {
    const isNew = QM.configBankCode === null;
    const code = (document.getElementById('qm-cfg-code')?.value || '').trim().toLowerCase();
    const name = (document.getElementById('qm-cfg-name')?.value || '').trim();

    if (!code || !name) {
        alert('Code and Name are required.');
        return;
    }

    const aliases = _splitComma(document.getElementById('qm-cfg-aliases')?.value);
    const domains = _splitComma(document.getElementById('qm-cfg-domains')?.value);
    const irPages = (document.getElementById('qm-cfg-ir-pages')?.value || '')
        .split('\n').map(s => s.trim()).filter(Boolean);

    // Collect doc_naming
    const docNaming = {};
    document.querySelectorAll('.qm-config-doc-type input[data-doc-type]').forEach(input => {
        const docType = input.dataset.docType;
        const field = input.dataset.field;
        if (!docNaming[docType]) docNaming[docType] = { document_aliases: [], url_patterns: [] };
        docNaming[docType][field] = _splitComma(input.value);
    });

    // Remove empty entries
    for (const [dt, entry] of Object.entries(docNaming)) {
        if (entry.document_aliases.length === 0 && entry.url_patterns.length === 0) {
            delete docNaming[dt];
        }
    }

    const payload = {
        code: code,
        name: name,
        aliases: aliases,
        primary_domains: domains,
        ir_pages: irPages,
        doc_naming: docNaming,
    };

    try {
        const url = isNew ? '/api/config/banks' : `/api/config/banks/${encodeURIComponent(QM.configBankCode)}`;
        const method = isNew ? 'POST' : 'PUT';
        const res = await fetch(url, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert('Save failed: ' + (err.detail || res.statusText));
            return;
        }

        // Refresh bank list and rebuild bar
        const banksRes = await fetch('/api/config/banks').then(r => r.json());
        QM.banks = banksRes;
        _buildBankBar();
        _highlightActiveBank();
        _closeConfigForm();
    } catch (err) {
        alert('Save failed: ' + String(err));
    }
}

/* ── Utility ──────────────────────────────────────────────────── */

function _esc(text) {
    const d = document.createElement('div');
    d.textContent = String(text ?? '');
    return d.innerHTML;
}

function _escAttr(s) {
    return String(s || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function _truncUrl(url) {
    if (!url) return '';
    try {
        const u = new URL(url);
        const path = u.pathname.length > 40 ? '...' + u.pathname.slice(-37) : u.pathname;
        return u.hostname + path;
    } catch {
        return url.length > 60 ? url.slice(0, 57) + '...' : url;
    }
}
