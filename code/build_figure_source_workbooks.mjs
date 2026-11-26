/** Build the 13 figure workbooks from the released or regenerated panel CSVs. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const archive = path.join(repo, 'source_data');
const output = path.join(repo, 'outputs', 'source_data');
fs.mkdirSync(output, {recursive: true});

const figures = [
  ['Fig1', 'Figure 1', 'Basin water budgets and modeled salt-export obligations'],
  ['Fig2', 'Figure 2', 'Field harvest, composite production and soil salinity'],
  ['Fig3', 'Figure 3', 'Hydraulic response, salt transport and audited water use'],
  ['Fig4', 'Figure 4', 'Withdrawals, depletion and baseline water budgets'],
  ['Fig5', 'Figure 5', 'Paired climate projections and two-boundary salt budgets'],
  ['ED_Fig1', 'Extended Data Figure 1', 'Baseline, blending assumptions and budget components'],
  ['ED_Fig2', 'Extended Data Figure 2', 'Water-budget sensitivity and groundwater comparisons'],
  ['ED_Fig3', 'Extended Data Figure 3', 'Evidence selection and residual diagnostics'],
  ['ED_Fig4', 'Extended Data Figure 4', 'Trial locations and area of applicability'],
  ['ED_Fig5', 'Extended Data Figure 5', 'Threshold and family-level moderator checks'],
  ['ED_Fig6', 'Extended Data Figure 6', 'Hydraulic responses by soil and amendment'],
  ['ED_Fig7', 'Extended Data Figure 7', 'Water-supply, ET and drainage-recovery scenarios'],
  ['ED_Fig8', 'Extended Data Figure 8', 'Observed district and catchment salt budgets']
];

// Use the same Python environment as the statistical models. Identifiers and
// leading zeros remain text even when the corresponding labels resemble numbers.
const writeBook = String.raw`import csv, json, math, re, sys
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

key, figure, description, files, destination = json.load(sys.stdin)
book = Workbook()
book.properties.creator = "Shu-Yang Yu, Jingsong Du and Shaobo Li"
book.properties.title = "Source Data | " + figure
contents = book.active
contents.title = "Contents"
contents.merge_cells("A1:D1")
contents["A1"] = book.properties.title
contents.merge_cells("A2:D2")
contents["A2"] = description
contents["A2"].font = Font(name="Aptos", color="225465", size=11)
contents.row_dimensions[2].height = 26
contents.append([])
contents.append(["Panel", "Worksheet", "Source table", "Data rows"])
contents.freeze_panes = "A5"

def header(sheet, index):
    for cell in sheet[index]:
        cell.font = Font(name="Aptos", size=11, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="225465")
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    sheet.row_dimensions[index].height = 30

def typed(value, column):
    if not value:
        return None
    if value in ("True", "False"):
        return value == "True"
    if not re.search(r"(?:^|_)(?:id|doi|code|identifier)(?:$|_)", column, re.I):
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value) and not re.match(r"^0\d", value):
            number = float(value)
            if math.isfinite(number):
                return number
    return value

header(contents, 1)
header(contents, 4)
for i, width in enumerate([14, 40, 64, 16], 1):
    contents.column_dimensions[get_column_letter(i)].width = width

for filename in files:
    file = Path(filename)
    stem = file.stem[len(key):]
    panel, _, label = stem.partition("_")
    panel = panel or ("Schematic" if key == "ED_Fig1" else "D-E" if "threshold_recovery" in stem else "A-C")
    title = (panel + " " + label.replace("_", " "))[:31].strip()
    if title in book.sheetnames:
        raise ValueError("Duplicate worksheet " + title)
    sheet = book.create_sheet(title)
    with file.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.reader(source)
        names = next(rows)
        sheet.append(names)
        widths = [len(name) for name in names]
        count = 0
        for count, raw in enumerate(rows, 1):
            if len(raw) != len(names):
                raise ValueError(f"{file.name} row {count} has {len(raw)} fields")
            values = [typed(v, name) for v, name in zip(raw, names)]
            sheet.append(values)
            for j, val in enumerate(values):
                if isinstance(val, str) and val.startswith("="):
                    sheet.cell(count + 1, j + 1).data_type = "s"
                if count < 120:
                    widths[j] = max(widths[j], len(str(val if val is not None else "")))
    header(sheet, 1)
    sheet.freeze_panes = "A2"
    for i, width in enumerate(widths, 1):
        sheet.column_dimensions[get_column_letter(i)].width = max(15, min(60, width + 2))
    contents.append([panel, title, file.name, count])
dest = Path(destination)
temp = dest.with_suffix(".partial.xlsx")
book.save(temp)
temp.replace(dest)
`;
for (const [key, figure, description] of figures) {
  const names = fs.readdirSync(archive).filter(n => n.startsWith(key) && n.endsWith('.csv')).sort();
  if (!names.length) throw new Error('Missing panel sources for ' + figure);
  const files = names.map(n => fs.existsSync(path.join(output, n)) ? path.join(output, n) : path.join(archive, n));
  const dest = path.join(output, 'Source_Data_' + key + '.xlsx');
  execFileSync(process.env.PYTHON || 'python3', ['-c', writeBook], {
    input: JSON.stringify([key, figure, description, files, dest]), stdio: ['pipe', 'inherit', 'inherit']
  });
  process.stdout.write(figure + ' | ' + names.length + ' panel tables\n');
}

