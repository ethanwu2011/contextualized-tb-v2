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
* Horsburgh et al. Treatment of tuberculosis. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/26605929/).
* Nunn et al. Role of co-trimoxazole prophylaxis in reducing mortality in HIV infected adults being treated for tuberculosis. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/18617486/).
* Ahmad et al. Treatment correlates of successful outcomes in pulmonary multidrug-resistant tuberculosis: an individual patient data meta-analysis. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/30215381/).
* Cevik et al. Bedaquiline-pretomanid-moxifloxacin-pyrazinamide for drug-sensitive and drug-resistant tuberculosis (SimpliciTB). [PubMed record](https://pubmed.ncbi.nlm.nih.gov/38768617/).

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

The cell-level review table behind Figure 4c and 4d ships with the code at
`figures/source_data/fig4_literature_validation.csv`, and `figures/pptx_export.py` reads it from
there. It holds the 23 cells' coefficient contrasts, P values, review states and sources, the same
values Figure 4 prints, and no patient rows or counts. Its column definitions and disclosure review
are in `figures/source_data/README.md`, and `scripts/scan_public.sh` pins its hash. The restricted
clinical data are not in the repository.
