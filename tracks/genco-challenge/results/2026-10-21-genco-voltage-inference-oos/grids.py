"""Grid registry for the out-of-sample |V| confirmation (candidate 9). UNAUDITED."""
from pathlib import Path
EXT = Path(__file__).resolve().parents[4] / "data/external"
TV, RS, DK = EXT / "genco-transfer-val", EXT / "genco-robust-start", EXT / "genco-ladder-datakit"
ROOT = EXT / "genco-voltage-oos"
PRIMARY = (
    [("RTE", TV / g, g) for g in ["case1888rte", "case2848rte", "case6470rte", "case6495rte", "case6515rte"]]
    + [("PEGASE", RS / g, g) for g in ["case2869pegase", "pglib_opf_case8387_pegase", "case9241pegase"]]
    + [("Polish", RS / g, g) for g in ["pglib_opf_case2383wp_k", "pglib_opf_case3012wp_k", "pglib_opf_case3375wp_k"]]
    + [("SimBench-HV", DK / g, g) for g in ["1-HV-mixed--0-no_sw", "1-HV-mixed--1-no_sw", "1-HV-mixed--2-no_sw", "1-HV-urban--0-no_sw", "1-HV-urban--1-no_sw", "1-HV-urban--2-no_sw"]])
SEEN = ([("GOC", DK / g, g) for g in ["pglib_opf_case500_goc", "pglib_opf_case793_goc", "pglib_opf_case2000_goc", "pglib_opf_case2312_goc", "pglib_opf_case2742_goc",
                                      "pglib_opf_case3022_goc", "pglib_opf_case3970_goc", "pglib_opf_case4020_goc", "pglib_opf_case4601_goc", "pglib_opf_case4619_goc",
                                      "pglib_opf_case4837_goc", "pglib_opf_case4917_goc", "pglib_opf_case9591_goc"]]
        + [("EPIGRIDS", DK / g, g) for g in ["pglib_opf_case5658_epigrids", "pglib_opf_case7336_epigrids"]])
ALL = PRIMARY + SEEN
FAMILY = {g: (fam, src) for fam, src, g in ALL}
N_SMALL_STAGED, N_BIG_STAGED, BIG_BUS = 102, 52, 3500   # staged scenarios -> test = int(0.99*staged)
