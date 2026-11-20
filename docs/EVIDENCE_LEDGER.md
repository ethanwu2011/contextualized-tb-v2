# Evidence ledger

This document records the source review used to interpret the drug-specific cells in the PATCH
figure. It is deliberately row-free. It contains no patient rows, cell-level estimates, identifiers,
or fitted artifacts.

## Scope

The review covers the 23 nominally flagged cells in the five-subtype by 26-drug grid. The review was
performed after the model fit and is interpretive rather than confirmatory. A flagged cell is not
treated as a treatment recommendation.

Each cell is assigned one of four review states.

* Concordant background means that the direction or clinical setting is compatible with an
  established source.
* Confounding by indication means that a recorded association could reflect who received the drug,
  the resistance pattern, or the care setting.
* Discordant as coded means that the recorded variable or drug token needs a measurement or
  pharmacology caveat before it can be compared with a source.
* Novel hypothesis means that no directly comparable source was identified in the scoped review.

These states describe evidence context. They do not establish causality, efficacy, or safety.

For the Figure 4 display, the legacy table values confirmed, confounded, discordant, and novel map to
Concordant background, Confounding by indication, Discordant as coded, and Novel hypothesis,
respectively. These are display labels only.

## Sources checked

The following primary or authoritative records were checked for identity, title, and bibliographic
details. The links are included so a reader can repeat the source check.

* WHO. *Global tuberculosis report 2023*. [WHO publication record](https://www.who.int/publications/i/item/9789240083851).
* WHO. *Global tuberculosis report 2022*. [WHO publication record](https://www.who.int/publications/i/item/9789240063129).
* Nahid et al. ATS, CDC, and IDSA guidelines for treatment of drug-susceptible tuberculosis. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/27516382/).
* Nahid et al. Official ATS, CDC, and IDSA clinical practice guidelines for drug-resistant tuberculosis. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/31729908/).
* Rosenthal et al. The TB Portals: an open-access, web-based platform for global drug-resistant-tuberculosis data sharing and analysis. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/28904183/).
* Samuels et al. Molecular detection of drug resistance in *Mycobacterium tuberculosis*. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/29563561/).
* Conradie et al. Treatment of highly drug-resistant pulmonary tuberculosis. [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC9490302/).
* Cegielski et al. Aminoglycosides and capreomycin in the treatment of multidrug-resistant tuberculosis: individual patient data meta-analysis of 12 030 patients from 25 countries, 2009-2016. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/33124668/).
* van der Walt et al. Treatment outcomes and adverse drug effects of ethambutol, cycloserine, and terizidone for the treatment of multidrug-resistant tuberculosis in South Africa. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/33046491/).
* Fâcă et al. Nutritional deficiencies and management in tuberculosis. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/40507146/).
* Sambarey et al. Integrative analysis of multimodal patient data and tuberculosis prognosis. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/38357663/).

## Interpretation rules

The review uses the recorded resistance class and drug token exactly as defined in
`docs/DATA_DICTIONARY.md`. Rifampicin harmonization in the external cohort is a class-based proxy,
not an assertion that the two registries measure an identical susceptibility assay. Cycloserine and
terizidone remain separate recorded variables because the pipeline does not map the prodrug to its
active parent. A source that discusses one variable is therefore not automatically evidence for the
other.

The source review is post hoc and was not blinded to the model output. It is included to make the
interpretive step auditable, not to replace an independent replication or a prospective trial.

## Reproduction

The figure export reads the locally generated aggregate file
`results/tables/fig4_literature_validation.csv`. The public repository does not contain that file or
the restricted clinical data. Researchers with an approved data-use agreement can regenerate the
aggregate table with the analysis workflow, review it against this ledger, and run
`bash scripts/scan_public.sh <PUBLIC_DIR>` before sharing any derived artifact.
