# Citation Audit for SMILE² AAAI Draft

Last updated: 2026-07-20

## Red-line rule

Citation metadata is treated as part of the scientific claim. A BibTeX entry is not considered safe until title, author order, venue, year, pages, DOI/arXiv ID, and publication type have been checked against authoritative sources.

## Strict verification loop

For each cited entry:

1. **Locate the canonical source.** Prefer, in order:
   - official GitHub repository README BibTeX if the paper provides code;
   - official proceedings/publisher page: CVF/OpenAccess, IEEE Xplore DOI metadata, ACM DL/author official page, NeurIPS/OpenReview/AAAI/OJS, arXiv for preprints;
   - DBLP/Crossref/PubMed/Google Scholar only as secondary cross-checks;
   - paper PDF title page when metadata pages disagree.
2. **Compare fields.** Check title spelling/case, all authors and order, venue/proceedings/journal, year, volume/number/pages, DOI/arXiv ID, and URL.
3. **Resolve conflicts conservatively.** If two sources disagree, mark `needs visual/PDF check`; do not silently choose the convenient source.
4. **Patch both bib copies.** Update `paper/aaai2027/smile2_aaai2026.bib` and `paper/feedback/references.bib` together.
5. **Compile after changes.** Run LaTeX and check that no citation/reference warnings appear.
6. **Record evidence.** Add source URL and short note here.

## Current active citation scope

The current English draft cites 40 unique keys. This file tracks the active cited set first; uncited candidate entries can be audited later if they are promoted into the paper.

## Verified / corrected in this pass

| Key | Status | Author/title action | Authoritative evidence | Notes |
|---|---|---|---|---|
| `choi2017kaist` | verified-fixed | Corrected authors to Inchang Choi, Daniel S. Jeon, Giljoo Nam, Diego Gutierrez, Min H. Kim; added DOI. | KAIST VCLab official page: https://vclab.kaist.ac.kr/siggraphasia2017p1/index.html | Previous bib wrote Daniel S. Kim and missed Giljoo Nam. |
| `hu2022hdnet` | verified | Matches CVF title, authors, CVPR 2022, pages 17542--17551. | CVF OpenAccess: https://openaccess.thecvf.com/content/CVPR2022/html/Hu_HDNet_High-Resolution_Dual-Domain_Learning_for_Spectral_Compressive_Imaging_CVPR_2022_paper.html | CVF page also links official GitHub. |
| `wang2024vheat` | verified-fixed | Converted to `@inproceedings`; added CVPR 2025 pages and CVF URL. | CVF OpenAccess: https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Building_Vision_Models_upon_Heat_Conduction_CVPR_2025_paper.html | Key name remains `wang2024vheat` for compatibility, but metadata is CVPR 2025. |
| `luo2025dsmt` | verified | Title/authors/journal/pages/DOI checked; added IEEE document URL. | PubMed: https://pubmed.ncbi.nlm.nih.gov/40198285/ ; DOI metadata mirror: https://colab.ws/articles/10.1109%2Ftip.2025.3556520 | IEEE page may require JS; PubMed/DOI metadata are consistent. |
| `wang2025s2transformer` | provisional-fixed | Removed duplicated `Xin Y. Yuan`; kept single `Xin Yuan`; added DOI and IEEE URL. | DOI metadata mirror: https://colab.ws/articles/10.1109%2Ftpami.2025.3543842 ; secondary cross-checks from author/DBLP-style listings | DOI mirrors can duplicate `Xin Yuan` as `Xin Y. Yuan`; keep one author unless final IEEE PDF proves otherwise. |
| `shu2026waveformer` | verified-fixed | Converted arXiv-only entry to AAAI 2026 proceedings metadata; added volume, number, pages and DOI. | AAAI OJS: https://ojs.aaai.org/index.php/AAAI/article/view/39737 | Key remains unchanged for compatibility. |
| `meng2020gapnet` | verified-fixed | Corrected authors to Ziyi Meng, Shirin Jalali, Xin Yuan; added arXiv metadata. | arXiv: https://arxiv.org/abs/2012.08364 | Previous bib incorrectly included Zhengjue Yu and Kun Xu. |
| `meng2020tsanet` | verified | Added ECCV DOI/URL; title/authors/pages checked. | Springer: https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12 | End-to-end compressive spectral imaging baseline. |
| `cai2022mst` | verified | Added CVPR DOI/URL; title/authors/pages checked. | CVF: https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Mask-Guided_Spectral-Wise_Transformer_for_Efficient_Hyperspectral_Image_Reconstruction_CVPR_2022_paper.html | Major E2E transformer baseline. |
| `cai2022cst` | verified | Added ECCV DOI/URL; title/authors/pages checked. | Springer: https://link.springer.com/chapter/10.1007/978-3-031-19790-1_41 | Major E2E transformer baseline. |
| `cai2022mstpp` | verified-fixed | Corrected pages to 744--754; added DOI. | DBLP: https://dblp.org/rec/conf/cvpr/CaiLLWZPTG22 | CVPRW 2022 spectral reconstruction reference. |
| `cai2022dauhst` | verified | Added NeurIPS URL; title/authors/pages checked. | NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2022/hash/f621c2ead473ca36763696b712ffda01-Abstract-Conference.html | DU boundary reference. |
| `miao2019lambdanet` | verified | Added CVF URL; title/authors/pages checked. | CVF: https://openaccess.thecvf.com/content_ICCV_2019/html/Miao_l-Net_Reconstruct_Hyperspectral_Images_From_a_Snapshot_Measurement_ICCV_2019_paper.html | Quantitative baseline. |
| `huang2021dgsmp` | verified | Added CVF URL; title/authors/pages checked. | CVF: https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Deep_Gaussian_Scale_Mixture_Prior_for_Spectral_Compressive_Imaging_CVPR_2021_paper.html | Quantitative baseline. |
| `zhang2024dpu` | verified | Added CVF URL and CVPR DOI; title/authors/pages checked. | CVF: https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Dual_Prior_Unfolding_for_Snapshot_Compressive_Imaging_CVPR_2024_paper.html | DU boundary reference. |
| `zhang2024ssr` | verified | Added CVF URL; title/authors/pages checked. | CVF: https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Improving_Spectral_Snapshot_Reconstruction_with_Spectral-Spatial_Rectification_CVPR_2024_paper.html | DU boundary reference. |
| `li2021fno` | verified | Checked title/authors/ICLR 2021 against ICLR official poster and DBLP; added OpenReview URL. | ICLR virtual poster: https://iclr.cc/virtual/2021/poster/3281 ; DBLP: https://dblp.org/rec/conf/iclr/LiKALBSA21 | OpenReview itself may show browser verification, so ICLR/DBLP are used as cross-checks. |
| `leethorp2022fnet` | verified | Added NAACL-HLT proceedings metadata, pages and DOI from ACL Anthology. | ACL Anthology: https://aclanthology.org/2022.naacl-main.319/ | Official BibTeX uses `Onta{\\~n}{\\'o}n`; local entry follows ACL spelling. |
| `guibas2022afno` | verified-fixed | Corrected title to official ICLR title and checked author order. | ICLR virtual poster: https://iclr.cc/virtual/2022/poster/6073 ; DBLP: https://dblp.org/rec/conf/iclr/GuibasMLTAC22 | OpenReview itself may show browser verification, so ICLR/DBLP are used as cross-checks. |
| `rao2021gfnet` | verified | Added NeurIPS volume and proceedings URL; title/authors checked. | NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2021/hash/07e87c2f4fc7f7c96116d8e2a92790f5-Abstract.html | Frequency-domain global filter related work. |
| `kong2023fftformer` | verified | Added CVPR pages, DOI and CVF URL; title/authors/pages checked against CVF, DOI cross-checked via DBLP. | CVF: https://openaccess.thecvf.com/content/CVPR2023/html/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.html ; DBLP: https://dblp.org/rec/conf/cvpr/KongDGLP23 | Frequency-domain restoration related work. |
| `wagadarikar2008cassi` | verified-fixed | Added DOI/Optica URL; checked Applied Optics volume/issue/pages. | Optica Publishing Group: https://opg.optica.org/ao/abstract.cfm?uri=ao-47-10-b44 | Single-disperser CASSI acquisition reference. |
| `gehm2007ddcassi` | verified-fixed | Added DOI; checked title/authors/Optics Express volume/issue/pages. | DOI/Optica: https://doi.org/10.1364/OE.15.014013 | Dual-disperser CASSI acquisition reference. |
| `yasuma2010cave` | verified-fixed | Added DOI/PubMed URL; checked IEEE TIP volume/issue/pages and author order. | PubMed: https://pubmed.ncbi.nlm.nih.gov/20350852/ | CAVE dataset/source paper. |
| `bioucasdias2007twist` | verified-fixed | Added PubMed URL while retaining DOI; checked IEEE TIP volume/issue/pages. | PubMed: https://pubmed.ncbi.nlm.nih.gov/18092598/ | TwIST optimization baseline. |
| `yuan2016gaptv` | verified-fixed | Standardized ICIP booktitle and added DBLP URL; DOI retained. | DBLP: https://dblp.org/rec/conf/icip/Yuan16 | GAP-TV optimization baseline. |
| `liu2018rank` | verified-fixed | Corrected formal publication year to 2019 while retaining key name and DOI; added source URL. | Duke Scholars metadata: https://scholars.duke.edu/publication/1355229 | This is DeSCI/rank-minimization baseline; DOI date is 2018 but TPAMI volume issue is 2019. |
| `kruse1993sam` | verified-fixed | Added ScienceDirect URL; DOI/title/authors/pages checked. | ScienceDirect DOI page: https://www.sciencedirect.com/science/article/pii/003442579390013N | SAM metric origin/reference. |
| `vaswani2017attention` | verified | Checked NeurIPS/NIPS 2017 title, authors, pages; kept NeurIPS proceedings URL. | DBLP: https://dblp.org/rec/conf/nips/VaswaniSPUJGKP17.html | Foundational attention reference. |
| `liu2021swin` | verified-fixed | Corrected ICCV page range to 9992--10002; checked authors and DOI. | IEEE DOI page: https://doi.org/10.1109/ICCV48922.2021.00986 ; metadata cross-check: https://ouci.dntb.gov.ua/en/works/4LgOn0L7/ | Previous local pages were off by 20 pages. |
| `chen2026phycosf` | verified-fixed | Corrected official title to remove extra “Spectral” before Super-Resolution; checked authors/arXiv/ICML note. | arXiv: https://arxiv.org/abs/2605.13583 | Preprint accepted by ICML 2026 according to arXiv comments; still preprint-style entry until proceedings metadata appears. |
| `simon2026scientifictheorydeeplearning` | verified-fixed | Normalized author names/order and arXiv DOI. | arXiv: https://arxiv.org/abs/2604.21691 | Narrative/theory support; preprint. |
| `karniadakis2021piml` | verified | Checked Nature Reviews Physics volume/pages/DOI and author order. | Nature DOI: https://doi.org/10.1038/s42254-021-00314-5 | Physics-informed ML survey. |
| `willard2022scientificknowledge` | verified-fixed | Changed to formal ACM CSUR issue year 2023, volume 55(4), article 66; kept compatibility key. | University of Minnesota metadata: https://experts.umn.edu/en/publications/integrating-scientific-knowledge-with-machine-learning-for-engine/ | ACM copyright/online history includes 2022, but issue metadata is 2023. |
| `weinan2017dynamicalsystems` | verified | Checked title, journal, volume/issue/pages and DOI. | Princeton profile: https://collaborate.princeton.edu/en/publications/a-proposal-on-machine-learning-via-dynamical-systems/ | Dynamical-systems view of machine learning. |
| `cai2023bisci` | verified | Checked NeurIPS 2023 title/authors/volume and official proceedings URL. | NeurIPS: https://papers.neurips.cc/paper_files/paper/2023/hash/788e086c07b8d6fa6b279df56e512312-Abstract-Conference.html | Efficient SCI related work. |
| `bioucasdias2012unmixing` | verified | Checked JSTARS title/authors/pages/DOI. | Open Remote Sensing metadata: https://openremotesensing.net/knowledgebase/hyperspectral-unmixing-overview-geometrical-statistical-and-sparse-regression-based-approaches/ | Classic HSI spectral mixture/redundancy reference. |
| `cui2024csnet` | verified | Checked IJCAI 2024 title/authors/pages/DOI. | IJCAI official page: https://www.ijcai.org/proceedings/2024/80 | Frequency-domain image restoration related work, not CASSI-specific. |
| `liu2025ssfan` | verified-fixed | Added MDPI URL; checked Remote Sensing volume/issue/article number/DOI and authors. | MDPI: https://www.mdpi.com/2072-4292/17/19/3382 | Recent E2E CASSI frequency-domain method. |
| `zeng2024sahsci` | verified | Checked ECCV 2024 title/authors and Springer DOI/pages. | ECVA: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8105_ECCV_2024_paper.php ; DBLP: https://dblp.org/rec/conf/eccv/ZengLCLPS24.html | Adapter/self-supervised SCI related work. |
## Active cited keys still pending strict audit

No active cited key is currently waiting for first-pass strict audit. Provisional entries below should still be re-checked before camera-ready if publisher metadata changes.

## Next audit batch recommendation

1. Re-check provisional entries before camera-ready if official publisher metadata changes.
2. When adding any new comparison method, audit the paper and checkpoint/source together before putting it into a table.













