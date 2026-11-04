"""Discover and run maintained entry points without importing them on listing."""
import argparse
import runpy
import sys
from project_bootstrap import ROOT, configure_paths

COMMANDS = {
    'translate-local-legacy': 'scripts/translation/run_title_translation_local.py',
    'translate-gemma-tiki': 'scripts/translation/run_tiki_gemma_supervised.py',
    'build-qwen-v6': 'scripts/notebooks/build_title_translation_qwen3_14b_awq_colab_v6.py',
    'build-kaggle-qwen': 'scripts/notebooks/build_title_translation_qwen3_14b_awq_kaggle_v6.py',
    'build-qa-repair': 'scripts/notebooks/build_qwen_qa_repair_notebooks.py',
    'build-comet-1688': 'scripts/comparison/build_1688_cometkiwi_notebook.py',
    'compare-1688': 'scripts/comparison/compare_1688_cometkiwi.py',
    'compare-tiki': 'scripts/comparison/compare_tiki_cometkiwi.py',
    'audit-release': 'scripts/audit/audit_release.py',
    'monitor-kaggle': 'scripts/monitoring/inspect_qwen_kaggle.py',
    'prepare-judge': 'scripts/judge/so_cap_du_lieu.py',
    'judge': 'scripts/judge/chay_judge.py',
    'merge-judged': 'scripts/judge/ghep_bo.py',
    'prepare-ft': 'scripts/ft_clean/xay_ft_clean.py',
    'merge-official': 'scripts/merge/gop_tiki_vao_1688.py',
}

def main():
    # Forward subcommand flags verbatim, especially --help, to its own parser.
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        command, forwarded = sys.argv[1], sys.argv[2:]
        configure_paths()
        path = ROOT / COMMANDS[command]
        sys.argv = [str(path), *forwarded]
        runpy.run_path(str(path), run_name='__main__')
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', *COMMANDS], nargs='?', default='list')
    args, forwarded = parser.parse_known_args()
    if args.command == 'list':
        for command, path in COMMANDS.items():
            print(f'{command:18s} {path}')
        return

if __name__ == '__main__':
    main()
