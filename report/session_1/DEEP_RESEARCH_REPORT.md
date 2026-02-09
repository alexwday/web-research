# Deep Research Report

**Query:** This research brief aims to explore the current state of nuclear fusion energy, with a specific focus on technological advancements and future projections regarding commercial power generation. The research should primarily highlight the latest developments in experimental reactors, providing an overview of the most promising technologies and methodologies currently being tested. The intended audience is the general public, so the information should be presented in an accessible manner while maintaining an intermediate level of technical detail to ensure clarity without oversimplification.

The research team should prioritize recent breakthroughs in nuclear fusion technology, including advancements in reactor designs such as tokamaks and inertial confinement systems. Additionally, it is essential to include insights into the challenges these technologies face, such as achieving net energy gain and the economic viability of fusion power. Future projections should be grounded in current trends and expert opinions, offering a balanced view of when commercial fusion power generation might realistically be achieved. The final output should be engaging and informative, aimed at educating the public on the significance of nuclear fusion as a potential energy source for the future.
**Generated:** 2026-02-07 15:56
**Words:** 1,958
**Sources:** 30

---

## Table of Contents

1. [Foundations and Recent Experimental Milestones in Fusion](#foundations-and-recent-experimental-milestones-in-fusion)
2. [Core Technologies, Enabling Advances and Engineering Bottlenecks](#core-technologies-enabling-advances-and-engineering-bottlenecks)
3. [Pathways to Commercialization: Economics, Policy, Timelines and Uncertainties](#pathways-to-commercialization-economics-policy-timelines-and-uncertainties)

---

## Executive Summary

The single most important finding is that fusion research has moved from theoretical promise to demonstrable scientific milestones—most notably inertial confinement “ignition” events and large‑scale tokamak engineering—but remains short of the systemic engineering, materials and economic breakthroughs required for commercial power generation. Laboratory ICF experiments at LLNL have produced target‑level energy gains (e.g., Dec 5, 2022: 2.05 MJ laser in → 3.15 MJ fusion yield, Q≈1.5; subsequent shots rising to ~8.6 MJ out from 2.08 MJ in reported in April 2025), yet these results expose critical system losses and reproducibility challenges: DOE analyses show that of ~1.9 MJ delivered to a shot only ~25 kJ reached the imploded DT fuel, and small capsule imperfections have prevented straightforward repetition. In magnetic confinement, ITER and legacy devices continue to validate plasma physics at scale (D–T ion temperatures ~150×10^6 K), while industrializing enabling hardware—ITER’s conductor program required ~2,015 km of manufactured conductor and ~700 t of Nb3Sn over ~8 years—illustrates the supply‑chain scale‑up needed for plants.

These findings matter because they clarify where scientific progress ends and engineering or economic risk begins. Target Q>1 demonstrates physics feasibility but not plant‑level gain: driver efficiency, repetition rate, tritium breeding, blanket heat capture, divertor heat‑exhaust (design loads of ~10–20 MW/m^2) and materials able to tolerate high neutron fluences are unresolved at plant scale. High‑field HTS magnets, advanced cryogenics, high‑power lasers/pulsed drivers, and digital tools (AI‑enhanced shot planning, digital twins) are enabling avenues, but multi‑stream readiness assessments (TRL/MRL/IRL/SRL) are required because component readiness can mask low producibility or integration readiness.

Economically and institutionally, two pathways are emerging: mission‑managed public programs (ITER/DEMO timelines with DEMO engineering in the 2030s and construction post‑2040; IFMIF‑DONES for materials testing) and an aggressive private sector pushing modular pilots. Private investment surged from ~$1.9B in 2021 to ~$9.7B in 2025; firms report median capital needs of ~$700M to reach pilot stage and estimate >$77B aggregate to commercialize first plants. The U.S. DOE roadmap (Oct 2025) targets “power on the grid by the mid‑2030s,” but optimistic company claims (some as early as 2028) diverge from conservative DEMO schedules, producing a realistic window of pilot‑scale demonstration in the 2030s and commercial fleets after 2040 unless prioritized engineering breakthroughs accelerate timelines.

The full report first details experimental foundations and recent milestones, then analyzes core technologies and engineering bottlenecks, and concludes with commercialization pathways, economics, policy implications and prioritized R&D actions needed to close the gap between scientific success and reliable, economical fusion power.

---

## Foundations and Recent Experimental Milestones in Fusion

### Fusion fuels and the meaning of "ignition"
- D–T (deuterium‑tritium) is the pragmatic near‑term fuel because it has the largest cross‑section at the lowest attainable temperatures: a single D–T reaction releases 17.6 MeV, of which ≈14.1 MeV (~80%) is carried by an uncharged neutron and ≈3.5 MeV by a confined alpha particle [3][5]. Typical magnetic‑confinement D–T targets (e.g., ITER) aim for ion temperatures of order 150×10^6 K; by contrast D–D or aneutronic fuels require far higher temperatures (D–D ≈400–500×10^6 K; p‑B ≈600×10^6 K) [2][3].  
- “Ignition” in fusion research refers to the plasma physics threshold where alpha‑particle self‑heating dominates external heating and the plasma becomes (partially) self‑sustaining. Operationally the community also uses Q — the ratio of fusion power produced to external heating power — as a metric. Demonstrating Q>1 at the target (scientific breakeven) is distinct from achieving engineering or plant‑level gain (QE>1) once driver inefficiencies, thermal conversion, and blanket/tritium systems are included [2][3][5].

### National Ignition Facility (ICF) — milestones, numbers and interpretations
- LLNL reported the first laboratory demonstration of inertial confinement fusion (ICF) ignition on Dec 5, 2022 (shot N221204): 2.05 MJ of UV laser energy delivered to the target produced 3.15 MJ of fusion yield (Q≈1.5 relative to laser energy on target) [3][1][5]. LLNL has since reported additional high‑yield events: July 2023 (2.05 MJ → 3.88 MJ), Oct 2023 (2.2 MJ → 3.4 MJ), Feb 2024 (~2.2 MJ → 5.2 MJ), and an April 2025 milestone reporting 8.6 MJ out from 2.08 MJ in (reported as >4× input at target) [4]. Facility peak powers for high‑energy shots are in the 400–500 TW range [3][4][1].  
- LLNL framed the Dec 5, 2022 result as “an historic inertial confinement fusion (ICF) experiment … achieving ignition and energy gain” [1][3]. DOE and the fusion community, however, explicitly distinguish that target‑level Q>1 is a critical scientific milestone while engineering/plant‑level gain (accounting for driver efficiency, repetition rate, target fabrication, thermal conversion and tritium breeding) remains unmet for commercial viability [5].

### Key technical caveats: coupling, reproducibility, and systems framing
- A crucial practical limit for ICF is coupling inefficiency: DOE analyses show that of ~1.9 MJ delivered by the laser in the Aug 2021 experiments, only ~25 kJ actually reached the imploded DT fuel — illustrating very large driver→fuel losses and the difference between “target Q” and a net power plant energy balance [5]. LLNL also documents that tiny imperfections in capsule fabrication were responsible for failed repeats after the Aug 2021 threshold shot: “What was a bit surprising is the level of perfection that is required from the target” [1].  
- LLNL is using ML/AI (CogSim and transfer‑learning workflows) and higher‑fidelity computation to improve shot planning and reproducibility; the lab also launched an internal IFE initiative and DOE workshops reported a surge in private investment (~$4.7B in the last decade) and IFE‑directed private funds (~$180M in the past two years) in response to ignition progress [1][5].

### What the experiments prove — and do not — for commercial power
- The NIF results demonstrate, for the first time in the laboratory, target‑level ignition (Q>1) in ICF and show that multi‑MJ fusion yields are experimentally achievable. They do not, by themselves, demonstrate a viable power plant: driver energy and repetition‑rate inefficiencies, target mass‑manufacture and injection, debris/optics survivability, blanket neutron capture and tritium‑breeding (TBR>1 target ≈1.1–1.2), and thermal‑to‑electric conversion must all be solved to reach plant QE>1 and sustained electricity production [3][5].  
- Magnetic‑confinement programs (ITER, JET, EAST, Wendelstein 7‑X) and private magnetic‑concept demonstrators play complementary roles in that systems‑level engineering (steady‑state operation, superconducting magnets, divertors, tritium blankets) is primarily being addressed there; those topics and readiness comparisons are treated in the next section, Core Technologies, Enabling Advances and Engineering Bottlenecks.

---

## Core Technologies, Enabling Advances and Engineering Bottlenecks

### Confinement concepts and readiness framing
Fusion readiness cannot be captured by a single number. The IAEA states fusion “remains in R&D and demonstration stages (not commercially viable)” and explicitly calls for a fusion‑specific TRL framework that tracks multiple TRL “streams” (materials, manufacturing, instrumentation, software, systems, device) because single TRL scores “obscure cross‑stream shortfalls” [13]. DoD and other bodies have codified complementary readiness metrics (TRL = 9‑level scale; MRL = 10‑level scale) and, most recently, the DoD TRA Guidebook (Feb 2025) formalizes TRL/MRL/IRL/SRL constructs and templates for assessments [14]. Scholarly reviews echo the need to augment TRL with manufacturability and integration measures (IRL/SRL, AD2/ITI) because complex systems can show high component TRL but low producibility (MRL) or integration readiness [12][5].

No authoritative, device‑level TRL/MRL table for the main confinement concepts (tokamak, stellarator, inertial confinement (ICF), magnetized target/MTF, field‑reversed configuration (FRC), z‑pinch) was found in the provided sources; IAEA provides component examples (breeding blanket, toroidal field coil) but not complete device scores [13]. Practically, tokamaks and ICF devices have the deepest experimental heritage (ITER/JET; NIF and other laser facilities), stellarators (Wendelstein 7‑X) have demonstrated important steady‑state physics advances, while MTF/FRC and z‑pinch concepts remain at earlier stages of integrated demonstration. Any credible readiness assessment must therefore apply multi‑stream TRL/MRL templates, specify required performance envelopes (pulse length, fluence, duty cycle) and explicitly score materials, manufacturing and fuel‑cycle streams per device [13][4].

### Enabling technologies: magnets, power, lasers and digital control
High‑field magnets and high‑current distribution are central enabling technologies. ITER’s magnets are LTS‑based (NbTi and Nb3Sn) with TF coils operated at ≈68 kA and peak fields ≈11.8 T, a central solenoid nominal current ≈45 kA (~13 T) and total manufactured conductor equivalent ≈2,015 km — reflecting an industrial scale‑up (≈700 t Nb3Sn produced over ~8 years, peak ≈100 t/yr vs pre‑ITER ~15 t/yr) [14]. ITER does not deploy HTS/REBCO for its main coils in the Baseline 2024 documents, which means next‑generation high‑field HTS technology remains an external enabler for compact‑high‑field tokamaks and future plant concepts [14].

Power‑handling hardware is also unprecedented: thick, actively cooled aluminium busbars sized to carry up to 70 kA, gyrotrons and neutral‑beam auxiliaries (SPIDER/MITICA) coming online, and plant‑scale cryogenics and pulsed electrical distribution already under commissioning [11][2][4]. For inertial confinement, high‑power lasers and pulsed‑power drivers remain the core enablers (see Foundations and Recent Experimental Milestones in Fusion), while private and national programs are advancing pulsed‑power concepts for MTF and z‑pinch approaches.

Digital technologies — large‑scale simulation, Scientific Computing & Data Centre infrastructure, AI for engineering and operations and nascent “digital‑twin” control systems — are being deployed to manage complexity, optimize design iteration and support commissioning [11][2].

### Primary engineering bottlenecks
Materials and plasma‑facing systems remain the top plant‑scale challenges. ITER moved to tungsten first‑wall panels and a full‑tungsten divertor to reduce operational risk, but tungsten brings its own issues (brittleness, joining, transient tolerance) and divertor cooling must handle quasi‑steady heat fluxes of ~10–20 MW/m² — an engineering load “unprecedented in conventional engineering components.” The tokamak divertor ring comprises 54 cassette assemblies with surface‑contour tolerances of ±250 μm over 1.5 m, imposing exacting manufacturing requirements [14]. Fabrication lessons from ITER (weld‑induced ferrite up to ~10%, microhardness rises up to +40.6%, shape deviations up to 0.41 mm) illustrate microstructural and dimensional risks that cascade into schedule impacts [13].

Tritium breeding and fuel‑cycle logistics remain unresolved at plant scale. Under Baseline 2024, neutron fluence to Test Blanket Modules during ITER’s initial DT phase is predicted to be <1% of some TBM objectives, provoking proposals (embedded electrical heaters) to exercise tritium release systems without full DT fluence [13]. Neutron damage, component life under high‑fluence irradiation, remote maintenance and supply‑chain scaling for critical materials (Nb3Sn, special alloys, tungsten manufacture and joinery) are concrete MRL‑level risks that will drive CAPEX and deployment timelines [14].

Finally, readiness metrics must be applied carefully: TRL assessments are subjective unless tightly scoped (single‑run vs multi‑condition tests), and fusion requires multi‑stream TRL/MRL assessments to avoid optimistic bias from isolated component maturity [15][3][2]. As explored in the next section (Pathways to Commercialization), these technical and supply‑chain realities will be decisive for cost, schedule and regulatory pathways to a commercial fusion power plant. [13][4][1]

---

## Pathways to Commercialization: Economics, Policy, Timelines and Uncertainties

### Commercialization pathways: public, private and hybrid models
Two distinct but overlapping pathways to commercialization are emerging. The public‑led route centers on ITER and the EU DEMO roadmap: the EU lists an ITER allocation of €5.61 billion for 2021–2027 and an ITER operation baseline in 2034, with DEMO engineering design slated for the 2030–2040 window and construction projected after 2040 [21][3]. Complementary infrastructure such as IFMIF‑DONES (materials test neutron source in Granada) is under construction to qualify materials for these timelines [21]. Parallel to this, a vigorous private‑sector pathway—represented by firms like Commonwealth Fusion Systems, Helion, Tokamak Energy, General Fusion and others—has attracted large venture capital flows (private investment grew from ~$1.9B in 2021 to ~$9.7B in 2025) and is pursuing faster, modular pilot routes that emphasize manufacturability and rapid iteration [22][3].

#### Public–private partnerships and governance
European institutions have mobilized to bridge these models: a Commission‑led Fusion Expert Group (kicked off 26 June 2024) and a Coordination & Support Action (CSA) launched Jan 2025 aim to prepare a co‑programmed public–private partnership (PPP) and industry structures over two years [21]. Political messaging frames fusion as a strategic, long‑term complement to renewables—“safe, clean, strategic” in the Commission’s language—and industry bodies are calling for mechanisms (EIC, PPPs) to channel private capital into deployment [23][1][4]. Academic stakeholders caution that PPPs must protect low‑TRL fundamental research and demand rigorous external evaluation when public funds go to private actors [25].

### Economics, business models and grid integration
Concrete CAPEX and LCOE estimates remain largely unpublished in the public summaries reviewed; industry narratives stress minimizing capital cost through repeatable manufacturing (HTS magnets, capacitor banks, pulsed systems) and supply‑chain “seeding” programs [23][4]. Surveyed firms report a median capital need of ~$700M per company to reach pilot‑plant stage and estimate aggregate capital >$77B to commercialize first plants—figures that underscore heavy upfront investment and financing needs [22]. Early commercial signals include offtake agreements: Helion’s PPA with Microsoft for at least 50 MWe and CFS arrangements with Google, plus announced site selections and repurposing of retired fossil sites, indicating utility and corporate interest in early grid integration [22][4]. Workforce projections anticipate >4,600 direct employees today, expanding toward ~18,200 when pilot fleets are operational [22].

### Timeline scenarios: optimism vs. structured roadmaps
Private firms and some media project pilot or commercial milestones in the 2030–2035 window, with a few company claims as aggressive as electricity sales by 2028 (Helion) [22][3]. The U.S. DOE Roadmap—released October 16, 2025—targets “power on the grid by the mid‑2030s,” backed by detailed milestones, an Office of Fusion Energy & Innovation, and a call for a “stable, predictable, and right‑sized regulatory framework” [24]. This produces a split between optimistic company timelines and more conservative, mission‑managed roadmaps balanced by public infrastructure timelines such as DEMO after 2040 [21][4].

### Prioritized R&D, policy actions and outstanding controversies
- Prioritized R&D: materials qualification (neutron‑flux testing), heat‑exhaust/divertor solutions, tritium breeding and integrated fuel‑cycle engineering; supply‑chain scale for HTS magnets and pulsed systems (as explored in Core Technologies, Enabling Advances and Engineering Bottlenecks) [see adjacent section].  
- Policy actions: create PPP legal frameworks, seed industrial supply chains, establish export‑control/nonproliferation rules adapted to fusion, and implement “right‑sized” regulation with predictable funding streams [21][4][5].  
- Controversies: divergent budget tallies for the next Euratom cycle (Science|Business €9.8B vs. FIA €6.7B interpretations), tensions between rapid industrial engagement and protecting low‑TRL public research, and the public absence of transparent CAPEX/LCOE benchmarks that would ground investor and utility planning [22][4][5]. As one industry synthesis put it, “Fusion no longer seems like a technology perpetually 20 years away—it is an energy solution actively being engineered, tested, and piloted for commercialization in this decade” [22]; concurrent academic statements warn that “one of the most urgent issues…is the underfunding of core research tasks” [25]. These competing claims will shape whether commercial fusion arrives on the accelerated timelines proponents forecast or follows the longer public‑infrastructure arc.

---

## Conclusion

The body of evidence assembled in this report presents a coherent if cautious picture: fusion has moved decisively past proof‑of‑principle and into a phase where scientific landmarks (notably ICF target‑level ignition) coexist with persistent, system‑level barriers to commercial electricity. Across confinement approaches—tokamaks, stellarators, ICF and emerging magnetized or pulsed concepts—the same underlying pattern recurs. Breakthroughs in isolated subdomains (laser coupling and target physics; high‑field magnet performance; stepped‑up materials tests) matter greatly, yet they do not by themselves bridge the larger engineering, supply‑chain and economic gaps that separate laboratory milestones from dispatchable, grid‑connected plants.

Three overarching themes structure this synthesis. First, a demonstration‑to‑deployment tension: scientific Q>1 at the target level is no longer the sole criterion for progress; what matters for commercialization is an integrated plant‑level energy and material balance that accounts for driver efficiency, repetition rate, thermal conversion and tritium breeding. Second, multi‑stream readiness: fusion readiness cannot be reduced to a single TRL. Device viability depends on concurrent maturity across magnets, plasma‑facing materials, fuel‑cycle engineering, manufacturing reproducibility and digital control, and deficits in any stream constrain overall progress. Third, governance and pathway plurality: public megaprojects and distributed private innovation are complementary but carry distinct risks—big‑science timelines and material test infrastructure on the one hand, rapid iteration and modular manufacturing on the other—requiring deliberate policy design to marshal both without eroding fundamental research.

These themes imply concrete practical actions. Policymakers should prioritize coordinated investment in integrated testing facilities (neutron sources, high‑heat‑flux divertor rigs) and in transparent, device‑level readiness assessments that guide funding and procurement. Investors and firms should shift from single‑metric narratives toward staged, de‑risking portfolios that fund supply‑chain scale‑up (notably HTS conductor manufacturing and pulsed‑power systems) and automated, high‑yield target fabrication for ICF. Grid operators and regulators must begin preparatory work on interconnection models, licensing pathways that accommodate iterative pilot plants, and workforce pipelines that blend materials, cryogenics, plasma physics and digital engineering skills.

To translate these priorities into research programs, the field needs targeted, actionable studies. First, develop and publish a standardized, multi‑stream TRL/MRL/IRL framework applied across major device classes, with explicit performance envelopes (pulse length, duty cycle, fluence) and pass/fail criteria for materials and manufacturing. Second, execute coordinated accelerated irradiation campaigns using IFMIF‑DONES and complementary surrogate platforms to generate high‑fluence data for candidate divertor and blanket materials, coupled with validated embrittlement and jointing studies for tungsten and liquid‑metal concepts. Third, invest in demonstrator programs for plant‑level energy accounting: couple high‑Q target experiments with realistic driver efficiency models and thermal conversion chains to quantify QE and identify dominant loss reductions needed for economic viability. Fourth, fund pilot manufacturing lines for HTS/REBCO conductor and high‑precision ICF targets to produce cost and yield curves and to stress‑test supply chains. Fifth, establish standardized digital‑twin testbeds that integrate AI shot planning, diagnostics, and control with transparent validation protocols to accelerate reproducibility.

If these integrative research trajectories are pursued in parallel with prudent, evidence‑based policy and financing mechanisms, the path from memorable laboratory milestones to reliable commercial fusion becomes far more tractable. The pace of that transition will hinge less on one dramatic experiment than on disciplined systems engineering, manufacturability, and governance that close the persistent gap between physics demonstrations and plant reality.

---

## Glossary

**Engineering gain (QE or plant-level gain)**: Ratio of electrical (or usable) energy output of an entire fusion power plant to the total energy input to the plant, including driver inefficiencies; QE>1 required for commercial electricity production.

**Ignition**: Provisional: the condition where alpha‑particle self‑heating is sufficient to maintain plasma temperature without external heating. Precise technical thresholds (e.g., for Q or fusion power fraction from alphas) require authoritative citation.

**MRL (Manufacturing Readiness Level)**: A 10‑level scale describing maturity of manufacturing processes, supply chains and producibility; used to evaluate ability to produce technology at scale and cost/quality readiness.

**Q (fusion gain)**: Provisional: ratio of fusion power produced to external heating power delivered to the plasma (plasma‑level Q). Note: engineering/plant definitions differ (see task above) — authoritative definition to be obtained from DOE/IAEA.

**Q (target-level gain)**: Ratio of fusion energy produced by a target to the driver energy delivered to the target (Q>1 = scientific breakeven/ignition in many ICF contexts).

**TRL (Technology Readiness Level)**: A 9‑level scale (TRL 1–9) describing maturity from basic principles (1) to proven operational systems (9); originally developed at NASA and widely adopted for technology assessments.

---

## Bibliography

1. **National Ignition Facility - 2022 - Annual Report** [Academic]
   https://annual.llnl.gov/fy-2022/national-ignition-facility-2022

2. **Cross-Section Sensitivity of the D-T Fusion Probability and ...** [Academic]
   https://www.osti.gov/servlets/purl/4215203

3. **Achieving Fusion Ignition** [Academic]
   https://lasers.llnl.gov/science/achieving-fusion-ignition

4. **The magic cocktail of deuterium and tritium**
   https://www.iter.org/node/20687/magic-cocktail-deuterium-and-tritium

5. **Delivering laser performance conditions to enable fusion ...** [Academic]
   https://www.sciencedirect.com/science/article/abs/pii/S1574181824000557

6. **Deuterium–tritium fusion**
   https://en.wikipedia.org/wiki/Deuterium%E2%80%93tritium_fusion

7. **NIF Sets Power and Energy Records** [Academic]
   https://lasers.llnl.gov/about/keys-to-success/nif-sets-power-energy-records

8. **What are D-D and D-T fusion cross-sections?**
   https://www.reddit.com/r/fusion/comments/ulgii4/what_are_dd_and_dt_fusion_crosssections/

9. **Inertial Fusion Energy 2022 FES Basic Research Needs ...** [Academic]
   https://science.osti.gov/-/media/fes/pdf/workshop-reports/2023/IFE-Basic-Research-Needs-Final-Report.pdf

10. **Scalable Chrysopoeia via (n,2n) Reactions Driven by ...**
   https://www.marathonfusion.com/alchemy.pdf

11. **TRLs and MRLs in 3D Printing**
   https://www.botfactory.co/blog/what-s-new-at-botfactory-1/post/trls-and-mrls-in-3d-printing-260?srsltid=AfmBOoq9EIzU9fm8LAPEsvde3C8KV7R2j0XDr3RIPwt_AaFCUfV48Lob

12. **FINANCIAL REPORT**
   https://www.iter.org/sites/default/files/media/2025-11/rapport-financier-iter-2024-web.pdf

13. **Contextual Role of TRLs and MRLs in Technology ...** [Academic]
   https://www.osti.gov/servlets/purl/1002093

14. **ANNUAL REPORT**
   https://www.iter.org/sites/default/files/media/2025-11/exe-ra-2024-ok-web.pdf

15. **IAEA TECDOC SERIES**
   https://www-pub.iaea.org/MTCD/Publications/PDF/TE-2047web.pdf

16. **ITER progresses into new baseline** [Academic]
   https://www.sciencedirect.com/science/article/abs/pii/S0920379625001905

17. **Technology Readiness Assessment Guidebook**
   https://www.cto.mil/wp-content/uploads/2025/03/TRA-Guide-Feb2025.v2-Cleared.pdf

18. **PROGRESS OF ITER AND ITS IMPORTANCE FOR FUSION ...**
   https://conferences.iaea.org/event/392/papers/35583/files/13455-Manuscript_P.Barabaschi.ITER.FEC%202025.v4.pdf

19. **Tools for Technology Evaluation: TRLs | by Carly Anderson**
   https://medium.com/prime-movers-lab/tools-for-technology-evaluation-trls-11daec23689

20. **Regular Technical Reports**
   https://www.iter.org/scientists/iter-technical-reports/regular-technical-reports

21. **Toward an EU Fusion Strategy**
   https://iea.blob.core.windows.net/assets/98a960b8-ec25-4e85-8e88-7beddf6cabd5/ElenaRighi_EURATOM.pdf

22. **Global Nuclear Fusion Energy Market Report 2026-2046**
   https://finance.yahoo.com/news/global-nuclear-fusion-energy-market-080300909.html

23. **Euratom: €9.8 billion for nuclear research in the next EU ...**
   https://sciencebusiness.net/news/nuclear-fusion/euratom-eu98-billion-nuclear-research-next-eu-budget

24. **The State of the Fusion Energy Industry in 2025**
   https://www.peaknano.com/blog/the-state-of-the-fusion-energy-industry-in-2025

25. **In focus: Europe's road to fusion energy - European Commission**
   https://energy.ec.europa.eu/news/focus-europes-road-fusion-energy-2025-04-15_en

26. **Top 5 Fusion Companies to Watch in 2026**
   https://www.cleanenergy-platform.com/insight/top-5-fusion-companies-to-watch-in-2026

27. **European Commission Proposes Record €6.7 Billion ...**
   https://www.fusionindustryassociation.org/european-commission-proposes-record-e6-7-billion-euratom-budget-with-major-boost-for-fusion-energy/

28. **U.S. DOE releases Roadmap for fusion energy ...**
   https://www.hoganlovells.com/en/publications/us-doe-releases-roadmap-for-fusion-energy-commercialization

29. **2025 07 01 EU fusion strategy (KTH).pdf**
   https://www.stockholmtrio.org/download/18.3104e4ad197a2309b5a829/1751449346880/2025%2007%2001%20EU%20fusion%20strategy%20(KTH).pdf

30. **Fusion Energy Market 2025-2045: Technologies, Players, ...**
   https://www.idtechex.com/en/research-report/fusion-energy-market/1094
