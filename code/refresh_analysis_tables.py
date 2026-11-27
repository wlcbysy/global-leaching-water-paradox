"""Build threshold, monitoring and paired-scenario table entries from the formal analyses."""

from config import OUT_TABLES, table_input, prepare_outputs
import csv
import json
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from statistics import median

UNIT = 'dS m<sup>-1</sup>'


def number(value, places=2):
    return format(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP), 'f')


def estimate(row, field):
    val = number(row[field + '_dsm'])
    if field + '_ci95' in row:
        low, high = row[field + '_ci95']
        return f'{val} ({number(low)}–{number(high)})'
    return val


def row(*cells):
    return '| ' + ' | '.join(map(str, cells)) + ' |'


def table(headers, rows):
    return '\n'.join([row(*headers), row(*[':---'] * len(headers)), *[row(*record) for record in rows]])


def load_csv(name):
    with table_input(name).open() as source:
        return list(csv.DictReader(source))


def refresh_tables():
    data = json.loads(table_input('pillar2_armA_ecw_results_current.json').read_text())
    models = data['dose_response_absolute']
    harvest = models['field_continuous_harvest']
    cut = models['continuous_excl_ry_gt2']
    monitor = next(x for x in data['monitoring_checks'] if x['threshold'] == 'field_harvest')
    low, high = (monitor['bands'][k] for k in ('lower', 'upper'))
    def n(r):
        return f"{r['k']:,} ({r['n_clusters']})"
    groups = [
        ('Field harvest, continuous', 'field_continuous_harvest'),
        ('Harvest and shoot biomass, continuous', 'continuous'),
        ('Continuous, relative production ≤ 2', 'continuous_excl_ry_gt2'),
        ('Multi-year, all schedules', 'multiyear_ge2yr'), ('Pooled production', 'pooled'),
        ('Field production, all schedules', 'field_only'), ('Exact variances', 'exact_variance'),
        ('Exact or LSD-derived variances', 'exact_plus_lsd'), ('Harvest yield, all schedules', 'grain_only'),
        ('Water-use efficiency', 'wue'), ('Cyclic, unadjusted', 'cyclic'), ('Blended', 'blended')]
    rows = []
    detailed_rows = []
    for label, key in groups:
        r = models[key]
        detailed_rows.append((label, 'Maas–Hoffman', n(r), estimate(r, 'ecw_star_90pct')+' '+UNIT,
                              '80% at '+estimate(r, 'ecw_star_80pct')+' '+UNIT, 'Family bootstrap'))
        if key not in ('field_continuous_harvest','continuous','continuous_excl_ry_gt2','multiyear_ge2yr','pooled'):
            continue
        rows.append((label, n(r), estimate(r, 'ecw_star_90pct'), estimate(r, 'ecw_star_80pct'),
                     f"*t* {number(r['t_dsm'])}, *b* {number(r['b_per_dsm'],4)}"))
    for key in ('continuous',):
        r = data['logistic_sensitivity'][key + '_sigmoidal_p_ge_1']
        rows.append((key.capitalize()+' sigmoidal', n(r), estimate(r,'ecw_star_90pct'),
                     estimate(r,'ecw_star_80pct'), 'Shape = '+number(r['p_shape'],1)))
    for label,k in [('Saturated-paste EC','saturated_paste_ECe_only'),('1:5 extract EC','extract_1_5_only')]:
        r = data['soil_EC_absolute'][k]
        rows.append((label,n(r),'—','—',f"+50% at {estimate(r,'ecw_plus50pct')}, +100% at {estimate(r,'ecw_plus100pct')} {UNIT}. Control water EC {number(r['median_ctrl_ecw_dsm'])} {UNIT}."))
    rows.extend([('Seasonal soil salinity below threshold', f"{low['rows']} ({low['families']})", '—', '—',
                  f"Post/pre {number(low['median_post_pre'],3)}, {number(low['share_rising']*100,1)}% rising"),
                 ('Seasonal soil salinity above threshold', f"{high['rows']} ({high['families']})", '—', '—',
                  f"Post/pre {number(high['median_post_pre'],3)}, {number(high['share_rising']*100,1)}% rising"),
                 ('Matched seasonal soil salinity',f"{monitor['matched_strata']} strata ({monitor['paired_families']})",'—','—',
                  f"High/low {number(monitor['family_median_high_low'])}, {monitor['families_higher']} families higher, *P* = {number(monitor['paired_sign_p'],3)}"),
                 ('Soil response by year and depth','467 (25)','—','—','Subsoil/topsoil slope ratio 1.26 in year 1 and 1.76 in year 3, interaction *P* = 0.551')])
    title = ('Extended Data Table 2. Irrigation-water thresholds retaining 90% and 80% of the fitted low-salinity response, and separate soil-salinity outcomes. '
             'Parentheses around thresholds give 95% family-bootstrap intervals. The piecewise model uses breakpoint *t* (dS m<sup>-1</sup>) and slope *b* (m dS<sup>-1</sup>). '
             'Seasonal monitoring ratios are dimensionless. The matched comparison uses a two-sided family sign test. Additional subsets and crop groups are reported in Supplementary Tables 3 and 4.')
    blocks = [title + "\n\n" + table(["Outcome and stratum", "Records (families)",
              "90% response EC<sub>w</sub> ("+UNIT+")", "80% response EC<sub>w</sub> ("+UNIT+")", "Other outcome or model parameter"], rows)]
    blocks.append("Additional Maas–Hoffman subsets\n\n"+table(["Stratum","Model","Records (families)","90% response","80% response","Interval"], detailed_rows))
    sar, ec = (data['family_level_sar'][key] for key in ('soil_SAR','soil_EC'))
    checks = [
        ("Field harvest", n(harvest), estimate(harvest, "ecw_star_90pct"), "90% response"),
        ("Continuous, relative production ≤ 2", n(cut), estimate(cut, "ecw_star_90pct"), "90% response"),
        ("Soil SAR", sar["n_families"], number(sar["spearman_r"], 3), "Rank correlation, P = " + number(sar["p"], 3)),
        ("Soil EC", ec["n_families"], number(ec["spearman_r"], 3), "Rank correlation, P = " + number(ec["p"], 3)),
        ("Paired soil EC below threshold", low["rows"], number(low["median_post_pre"], 3), "Median post/pre"),
        ("Paired soil EC above threshold", high["rows"], number(high["median_post_pre"], 3), "Median post/pre"),
    ]
    blocks.append("Field and monitoring checks\\n\\n" + table(["Outcome", "Records or families", "Estimate", "Definition"], checks))
    origin = load_csv('pillar3_threshold_and_recovery_current.csv')
    marker = 'Table 9, continued.'
    follow = marker+' Crop thresholds, ET estimates and marginal drainage recovery. Withdrawal and net depletion relief are in km<sup>3</sup> yr<sup>-1</sup>. Positive values indicate relief. The last column gives the number of initially exceeded basins returning within budget under caps and at the specified ET ratio.'
    r9=[(r['case'],number(r['ecw_dsm'],3),number(r['et_ratio'],4),number(float(r['marginal_leach_recovery'])*100,0)+'%',
         number(r['withdrawal_relief_km3_yr']),number(r['net_capped_km3_yr']),number(r['net_et_km3_yr']),
         r['rescued_capped']+' / '+r['rescued_et']) for r in origin]
    blocks.append(follow+'\n\n'+table(['Case','EC<sub>w</sub> ('+UNIT+')','ET ratio',
                     'Recovery','Withdrawal','Cap net','ET net','Rescued cap / ET'],r9))
    fut = defaultdict(list)
    for record in load_csv('pillar3_future_threshold_sensitivity_current.csv'):
        fut[record['scenario']].append(record)
    def span(values, med=False):
        low, high = number(min(values)), number(max(values))
        return f'{low}–{high}'+(f' [{number(median(values))}]' if med else '')
    def dif(group):
        return [float(r['zone3_et_mha'])-float(r['zone3_capped_mha']) for r in group]
    r11=[]
    for scenario, records in sorted(fut.items()):
        ref = [r for r in records if r['case']=='Composite reference']
        field = [r for r in records if r['case']=='Field harvest']
        prefix,year=scenario.split('_')
        name={'ssp126':'SSP1-2.6','ssp370':'SSP3-7.0','ssp585':'SSP5-8.5'}[prefix]+' '+year
        r11.append((name,span(dif(ref),True),span(dif(field),True),span([float(r['sal_arid_and_gw_mha']) for r in ref],True),
                    span([float(r['withdrawal_relief_km3_yr']) for r in records])))
    marker='Table 11, continued.'
    follow=marker+' Same-run ET-minus-cap Zone III differences (Mha), capped salinity–aridity and groundwater risk (Mha), and withdrawal relief across seven thresholds (km<sup>3</sup> yr<sup>-1</sup>). Ranges [medians] span five forcings per pathway–horizon; mixed-reference and field-harvest areas match because identical basins exceed budget.'
    block11=follow+'\n\n'+table(['Pathway and horizon','Paired area, mixed reference',
                       'Paired area, field harvest','Joint risk, mixed reference','Withdrawal range, all thresholds'],r11)
    blocks.append(block11)
    dest = OUT_TABLES / "analysis_tables.md"
    dest.write_text("\n\n".join(blocks)+"\n", encoding="utf-8")
    print(f"Wrote {dest.name}")


if __name__ == "__main__":
    prepare_outputs()
    refresh_tables()
