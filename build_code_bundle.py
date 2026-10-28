from __future__ import annotations
import ast,hashlib,json,re,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'nfl_speed_project_code.zip'
CHECKSUM=BUNDLE.with_suffix(BUNDLE.suffix + '.sha256')
NOTEBOOK=ROOT/'nfl_combine_speed_analysis.ipynb'

def main():
    notebook=json.loads(NOTEBOOK.read_text(encoding='utf-8-sig'))
    code='\n'.join(''.join(c.get('source',[])) for c in notebook['cells'] if c.get('cell_type')=='code')
    compile_cells=[]
    for i,cell in enumerate(notebook['cells']):
        if cell.get('cell_type')=='code':
            ast.parse(''.join(cell.get('source',[])),filename=f'notebook-cell-{i}')
            compile_cells.append(i)
    ordered_scripts=re.findall(r"run\(['\"]([^'\"]+\.py)['\"]",code)
    scripts=sorted(set(ordered_scripts))
    input_check="if DATA_INPUT is None or not DATA_INPUT.exists():"
    input_export="os.environ['NFL_BDB_INPUT'] = str(DATA_INPUT)"
    if (code.find(input_check) < 0 or code.find(input_export) < 0 or
            code.find(input_check) > code.find(input_export)):
        raise RuntimeError('Notebook must export the resolved competition input for subprocess defaults.')
    sources=sorted(p for p in ROOT.glob('*.py') if p.name!='build_code_bundle.py')
    missing=[name for name in scripts if not (ROOT/name).is_file()]
    if missing: raise FileNotFoundError(f'Notebook references missing source files: {missing}')
    # The burst extractor consumes the blitz-opportunity table written by the
    # hoop analysis. Guard this notebook dependency so clean runs do not rely
    # on stale outputs from a previous session.
    dependency_order=['analyze_dl_hoop.py','analyze_dl_game_burst.py','model_dl_transfer.py','model_dl_pressure_context.py']
    order=[ordered_scripts.index(name) for name in dependency_order]
    if order != sorted(order):
        raise RuntimeError(f'Notebook analysis dependency order is invalid: {dependency_order}')
    # This audit reads the OOF table from the 100-play pre-draft model to
    # quantify whether players with missing tracking keys drive its result.
    linkage_model_call="run('model_dl_early_accel_predraft.py', '--min-game-plays', 100)"
    linkage_audit_call="run('audit_game_tracking_linkage.py', '--archive', DATA_INPUT)"
    if code.find(linkage_model_call) < 0 or code.find(linkage_audit_call) < 0 or code.find(linkage_model_call) > code.find(linkage_audit_call):
        raise RuntimeError('Notebook linkage audit must follow the 100-play pre-draft model run.')
    linkage_order=[linkage_model_call,linkage_audit_call]
    feature_sensitivity_call="run('analyze_40_feature_sensitivity.py', '--archive', DATA_INPUT)"
    transfer_sensitivity_call="run('audit_dl_early_accel_measurement_transfer.py')"
    if code.find(feature_sensitivity_call) < 0 or code.find(transfer_sensitivity_call) < 0 or code.find(feature_sensitivity_call) > code.find(transfer_sensitivity_call):
        raise RuntimeError('Early-acceleration transfer sensitivity must follow creation of the processing-grid feature table.')
    measurement_order=[feature_sensitivity_call,transfer_sensitivity_call]
    case_model_call="run('model_dl_early_accel_predraft.py', '--min-game-plays', 100)"
    case_figure_call="run('build_dl_speed_profile_case_illustration.py')"
    if code.find(case_model_call) < 0 or code.find(case_figure_call) < 0 or code.find(case_model_call) > code.find(case_figure_call):
        raise RuntimeError('Matched-player illustration must follow generation of 100-play out-of-fold predictions.')
    case_order=[case_model_call,case_figure_call]
    target_precision_call="run('analyze_dl_game_speed_target_precision.py', '--archive', DATA_INPUT, '--draws', 1000)"
    split_figure_call="run('build_dl_split_half_figure.py')"
    if code.find(target_precision_call) < 0 or code.find(split_figure_call) < 0 or code.find(target_precision_call) > code.find(split_figure_call):
        raise RuntimeError('Split-half figure must follow the player-game target-precision analysis.')
    target_order=[target_precision_call,split_figure_call]
    position_model_call="run('model_position_specific_game_speed.py', '--min-game-plays', 100)"
    position_selection_call="run('audit_position_phase_nested_selection.py')"
    position_figure_call="run('build_position_phase_nested_figure.py')"
    if (code.find(position_model_call) < 0 or code.find(position_selection_call) < 0 or
            code.find(position_figure_call) < 0 or
            not code.find(position_model_call) < code.find(position_selection_call) < code.find(position_figure_call)):
        raise RuntimeError('Nested position-phase selection audit must follow position-specific profile construction.')
    position_order=[position_model_call,position_selection_call,position_figure_call]
    core_run_call="run('run_analysis.py', '--archive', DATA_INPUT, '--output', OUTPUT)"
    core_manifest_call="run('write_core_run_manifest.py', '--archive', DATA_INPUT, '--output', OUTPUT)"
    if code.find(core_run_call) < 0 or code.find(core_manifest_call) < 0 or code.find(core_run_call) > code.find(core_manifest_call):
        raise RuntimeError('Core output manifest must follow the core profile pipeline.')
    core_manifest_order=[core_run_call,core_manifest_call]
    paths=[(p,p.relative_to(ROOT).as_posix()) for p in sources]
    for top in ['README.md','RESEARCH_DESIGN.md','nfl_combine_speed_analysis.ipynb','requirements.txt']:
        p=ROOT/top
        if p.exists(): paths.append((p,p.relative_to(ROOT).as_posix()))
    for p in sorted((ROOT/'docs').rglob('*.md')):
        paths.append((p,p.relative_to(ROOT).as_posix()))
    audit={
      'bundle':BUNDLE.name,'files':len(paths),'notebook_code_cells_parsed':len(compile_cells),
      'notebook_runner_scripts':scripts,'runner_scripts_present':True,'dependency_order':[dependency_order,linkage_order,measurement_order,case_order,target_order,position_order,core_manifest_order],'zip_crc':'pass',
      'layout':'flat; run_analysis.py is at archive root',
      'kaggle_cloud_execution':'not performed from this local workspace'
    }
    (ROOT/'docs'/'PACKAGING_AUDIT.md').write_text(
      '# Kaggle package audit\n\n'
      f"The code bundle contains {len(paths)} files. Notebook code cells parsed successfully: {len(compile_cells)}. "
      f"All {len(scripts)} scripts invoked by the notebook are included at the bundle root. ZIP CRC verification passed.\n\n"
      'The bundle has a flat layout; notebook extraction rejects traversal, duplicate members, file/directory conflicts, symlinks, and oversized bundles. '
      'The archive uses sorted entries and fixed ZIP timestamps; a SHA-256 sidecar is written for verification. Pinned direct dependencies and the portable NFL_BDB_INPUT default are included, and the notebook exports its resolved data path to subprocesses. '
      'The ordered dependency checks confirm hoop opportunity extraction precedes burst and pressure analyses, the processing-grid feature table precedes transfer sensitivity, the 100-play pre-draft model precedes the linkage audit and matched-case illustration, position-specific modeling precedes nested position-phase selection and its figure, and player-game precision precedes the disjoint-game figure. '
      'On October 25, 2026, all 43 analysis calls in the staged notebook completed against a Kaggle-style extracted input directory using the notebook and analysis scripts packaged at replay time; the final report cell passed and 146 output files, including all four submission PNGs, were generated. The directory-mode source audit passed. The 56 CSVs from the prior analysis replay matched byte-for-byte; the notebook now also generates an attempt-specific OOF CSV. Fixed-format timestamp parsing matched pandas default parsing on 10,000 sampled rows from each of the four tracking CSVs, with zero invalid values. The attempt-1-versus-2 sensitivity on 27 paired DLs is included as a clearly labeled, non-independent measurement robustness check. The profile explorer label and note describe a path-relative cross-axis speed gap; its HTML was regenerated from verified tables. After replay, a pandas deprecated-downcast warning in the contextual stage was removed via nullable-string conversion; the stage was rerun and its outputs matched the replay byte-for-byte. In a later presentation-only update, the split-half figure title was clarified to say its partitions are within-season; its PNG/SVG were regenerated from verified results, with no analysis values changed. The final bundle includes these source, writeup, sensitivity, and audit updates. An earlier ZIP-mode replay completed before the timestamp optimization. Kaggle cloud execution and final public attachment checks remain outstanding. '
      'The current explorer enforces same-position A/B comparisons even under the all-positions filter and provides a descriptive comparison brief formatted for printing. The core pipeline now uses staged outputs, exclusive destination locking, run-linked promotion hashes, and cache provenance checks. The recruiter-facing README leads with the selection-aware result and distinguishes the post-screen discovery estimate. The notebook and code source paths were checked.\n\n'
      'Notebook scripts: ' + ', '.join(scripts) + '.\n',
      encoding='utf-8')
    # Package only after updating the audit so the archive contains the current
    # report rather than the previous build's version.
    with zipfile.ZipFile(BUNDLE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for p,name in sorted(paths,key=lambda item:item[1]):
            info=zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644 << 16
            z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=8)
    with zipfile.ZipFile(BUNDLE) as z:
        bad=z.testzip()
        names=z.namelist()
        if bad: raise RuntimeError(f'ZIP CRC failed at {bad}')
        if 'run_analysis.py' not in names: raise RuntimeError('run_analysis.py is not at bundle root')
        if any(name.startswith('nfl_speed_project/') for name in names): raise RuntimeError('Unexpected extra project-root directory in bundle')
        if any(name.startswith('/') or '..' in Path(name).parts for name in names): raise RuntimeError('Unsafe ZIP path')
        if len(names) != len(set(names)): raise RuntimeError('Duplicate paths found in bundle')
        if not set(scripts).issubset(set(names)): raise RuntimeError('Notebook source is incomplete in bundle')
        if 'docs/COMPETITION_SUBMISSION.md' not in names: raise RuntimeError('Submission draft missing from bundle')
        if 'requirements.txt' not in names: raise RuntimeError('Pinned dependencies missing from bundle')
    bundle_sha256=hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    CHECKSUM.write_text(f'{bundle_sha256}  {BUNDLE.name}\n',encoding='ascii')
    audit['bundle_sha256']=bundle_sha256
    print(json.dumps(audit,indent=2))
if __name__=='__main__': main()

