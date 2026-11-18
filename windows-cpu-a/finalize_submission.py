"""Validate the completed local official run and build the teacher-facing report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'project'
SESSION = PROJECT / 'sessions/formal'
sys.path.insert(0, str(PROJECT))
from adaptive.protocol import atomic_json, digest, parse_request, parse_response, weight_key
from adaptive.release import records, verify_manifest


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_specialization(batch,x,y,number,online,folder,session=None):
    import numpy as np
    from adaptive.structure_search import select
    from adaptive.base import configure,load_ensemble
    original=read(folder/'original-pool/candidates.json')
    count=original['role_counts']['exploit'];base=np.asarray(original['weights']).reshape(-1,21)
    seen={weight_key(w.tolist()) for w in x}
    configure();models=load_ensemble((session or SESSION)/'models'/f'v{number-1:02d}')
    from adaptive.acquisition import START_ROUND, compose
    if number >= START_ROUND:
        _, _, _, expected = compose(models, x, y, number, original['alpha'], online, seen, original)
        require(all(batch[key] == value for key, value in expected.items()),
                'Paired-strategy replay mismatch')
        return
    seen |= {weight_key(w.tolist()) for w in base}
    q,src,meta,report=select(models,x,y,number,batch['alpha'],online,seen,list(base[:count-6]))
    slots=list(range(count-len(q),count))
    require(batch['specialization']==dict(report,replacement_slots=slots),'Specialization report mismatch')
    expected=base.copy();sources=list(original['sources']);metadata=list(original['metadata'])
    for i,w,source,m in zip(slots,q,src,meta):expected[i]=w;sources[i]=source;metadata[i]=m
    require(np.array_equal(expected,np.asarray(batch['weights']).reshape(-1,21)),'Reserved selection/fallback mismatch')
    require(batch['sources']==sources and batch['metadata']==metadata,'Selection reasons mismatch')
    require(batch['alpha']==original['alpha'] and batch['base_alpha']==batch['alpha'] and batch['exploration_radius_multiplier']==1.,'Original radius changed')
    saved_pool=read(folder/'specialized-pool.json')
    from adaptive.structure_search import specialized_pool,transfer_pool
    pool,pm,banks=specialized_pool(x,y,number);tq,tm=transfer_pool(x,y,batch['alpha'])
    require(saved_pool['weights']==pool.reshape(-1,3,7).tolist() and saved_pool['metadata']==pm,'Sample pool mismatch')
    require(saved_pool['transfers']==tq.reshape(-1,3,7).tolist() and saved_pool['transfer_metadata']==tm,'Transfer pool mismatch')
    from adaptive.base import predict
    require(saved_pool['predictions']==predict(models,pool).T.tolist(),'Pool predictions mismatch')
    if len(tq):require(saved_pool['transfer_predictions']==predict(models,tq).T.tolist(),'Transfer predictions mismatch')


def verify_archive_manifest(archive):
    checksums = read(ROOT / 'archive-manifest.json')
    files = {p.relative_to(archive).as_posix() for p in archive.rglob('*') if p.is_file()}
    require(files == set(checksums), 'Archive file inventory changed')
    for name, expected in checksums.items():
        path = (archive / name).resolve()
        require(path.is_relative_to(archive.resolve()) and digest(path) == expected,
                'Archived artifact changed: ' + name)


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--archived', action='store_true', help='Verify the included completed experiment')
    parser.add_argument('--replay-selection', action='store_true',
                        help='Require exact replay of archived acquisition on this CPU')
    args = parser.parse_args()
    if args.replay_selection and not args.archived:
        parser.error('--replay-selection requires --archived')
    archive = ROOT / 'evidence'
    session = archive / 'formal' if args.archived else SESSION
    ledger_dir = archive if args.archived else PROJECT
    replay_selection = not args.archived or args.replay_selection
    if args.archived:
        verify_archive_manifest(archive)
    state = read(session / 'state.json')
    require(state['status'] == 'complete' and state['completed_rounds'] == 20,
            'The local formal run is not complete')
    require(state['backend'] == 'native' and state['attempted_calls'] == 20 and state['pending'] is None,
            'Official backend or call accounting mismatch')
    from submission import check
    check(require_executable=not args.archived)
    checked = verify_manifest()
    source = read(ROOT / 'SOURCE.json')
    config = read(session / 'config.json')
    require(config['backend'] == 'native' and config['initialization'] == 'fresh_fixed_seeds_original_only'
            and not config['historical_extra_labels_used'], 'Cold-start configuration mismatch')
    from adaptive.runner import source_fingerprint
    require(config['source_sha256'] == source_fingerprint(), 'Run source or strategy changed')
    ledger = read(ledger_dir / 'formal-budget.json')
    require(len(ledger['attempts']) == 20, 'Expected exactly twenty budget reservations')
    original = records()
    rows = read(session / 'online.json')
    require(len(rows) == 1000, 'Expected 1000 local official feedback rows')
    seen = {weight_key(r['weights']) for r in original}
    expected_rows = []
    summaries = []
    best = min(original, key=lambda r: r['R'])
    for number in range(1, 21):
        folder = session / 'rounds' / f'{number:02d}'
        request = parse_request(read(folder / 'request.json'))
        scores = parse_response(read(folder / 'stdout.json'), 50)
        batch = read(folder / 'candidates.json')
        visible=original+expected_rows
        if replay_selection:
            validate_specialization(batch, np.asarray([r['weights'] for r in visible]).reshape(-1, 21),
                                    np.asarray([r['R'] for r in visible]), number, rows[:50*(number-1)],
                                    folder, session=session)
        pairs={}
        for m in batch['metadata']:
            if 'pair_id' in m:pairs.setdefault(m['pair_id'],[]).append(m['pair_member'])
        require(all(sorted(v)==[0,1] for v in pairs.values()),'Learning pair incomplete')
        require(len(request) == 50 and request == batch['weights'], 'Candidate/request mismatch')
        require(read(folder / 'returncode.json')['returncode'] == 0, 'Official executable failed')
        attempt = ledger['attempts'][number-1]
        require(attempt['status'] == 'complete' and attempt['round'] == number
                and attempt['input_sha256'] == digest(folder / 'request.json'), 'Budget/request mismatch')
        expected_quota = (30, 20) if number <= 6 else (35, 15) if number <= 14 else (40, 10) if number <= 19 else (50, 0)
        require((batch['roles'].count('exploit'), batch['roles'].count('learn')) == expected_quota,
                'Sampling role quota mismatch')
        require(batch['prediction_model_version'] == number-1, 'Future model used for prediction')
        for index, (weights, score) in enumerate(zip(request, scores), 1):
            key = weight_key(weights)
            require(key not in seen, 'Duplicate candidate detected')
            seen.add(key)
            expected_rows.append(dict(id=f'r{number:02d}-{index:02d}', round=number,
                                      index=index, weights=weights, R=score, source=batch['sources'][index-1],
                                      role=batch['roles'][index-1], acquisition=batch['metadata'][index-1],
                                      predictions_before_feedback=batch['predictions'][index-1]))
        best = min([best, *expected_rows[-50:]], key=lambda r: r['R'])
        summary = read(folder / 'feedback.json')
        require(summary['batch_min_R'] == min(scores) and summary['cumulative_best_R'] == best['R'],
                'Progress summary mismatch')
        summaries.append(summary)
    for actual, expected in zip(rows, expected_rows):
        require(all(actual[k] == expected[k] for k in expected), 'Feedback order mismatch')
    saved = read(session / 'best.json')
    require(saved['R'] == best['R'] and saved['weights'] == best['weights']
            and saved['origin'] == best['id'] and saved['official_executable_used'], 'Winner mismatch')
    models_checked = 0
    for number in range(21):
        folder = session / 'models' / f'v{number:02d}'
        metadata = read(folder / 'training.json')
        visible = original + expected_rows[:50*number]
        x = np.asarray([r['weights'] for r in visible]).reshape(-1, 21)
        y = np.asarray([r['R'] for r in visible])
        with np.load(folder / 'training_samples.npz', allow_pickle=False) as arrays:
            require(np.array_equal(arrays['x'], x) and np.array_equal(arrays['y'], y),
                    f'Model v{number:02d} saw wrong training rows')
        expected_fp = hashlib.sha256(x.tobytes()+y.tobytes()+b'None').hexdigest()
        require(metadata['data_fingerprint'] == expected_fp and not metadata['old_models_loaded']
                and metadata['original_rows'] == 2048 and metadata['online_rows'] == number*50,
                'Training provenance mismatch')
        require([m['seed'] for m in metadata['models']] == [2026,2027,2028,2029,2030], 'Seed mismatch')
        for model in metadata['models']:
            require(digest(folder / f'seed{model["seed"]}.npz') == model['sha256'], 'Checkpoint hash mismatch')
            models_checked += 1
    diagnostic_checked = 0
    for folder in sorted((session / 'learning-audits').glob('after-*')):
        metadata = read(folder / 'training.json')
        for model in metadata['models']:
            require(digest(folder / f'seed{model["seed"]}.npz') == model['sha256'], 'Diagnostic hash mismatch')
            diagnostic_checked += 1
    require(models_checked == 105 and diagnostic_checked == 4, 'Expected 105 main and 4 diagnostic models')
    roles = {role:sum(r['role'] == role for r in rows) for role in ('exploit','learn')}
    require(roles == dict(exploit=710, learn=290), 'Total role quota mismatch')
    timing = read(session / 'timing-summary.json')
    guarded = set()
    if args.archived:
        original_validation = read(archive / 'verification/LOCAL_VALIDATION.json')
        require(original_validation['complete'] and original_validation['official_executable_used']
                and original_validation['executable_sha256'] == source['teacher_executable_sha256'],
                'Recorded original scorer provenance mismatch')
    else:
        for file in (PROJECT / 'access-audits' / session.name).glob('*.jsonl'):
            for line in file.read_text().splitlines():
                item = json.loads(line)
                if item.get('event') == 'guard_installed':
                    guarded.add(item['pid'])
                elif 'path' in item:
                    target = Path(item['path']).resolve()
                    require(target.is_relative_to(PROJECT.resolve()) or
                            any(v in ('.venv', '.tools') for v in target.parts), 'Foreign input audit')
        require(len(guarded) >= 3, 'Missing parent/spawned-worker isolation audit')
    validation = dict(guarded_processes=None if args.archived else len(guarded),structural_prior=True,specialization_verified=replay_selection,exploration_radius_multiplier=1.0,date=state['started_at'][:10], experiment='formal', strategy=source['strategy'], complete=True,
                      original_rows=2048, official_calls=20, official_feedback_rows=1000,
                      official_executable_used=True, executable_sha256=source['teacher_executable_sha256'],
                      source_commit=source['commit'], historical_labels_used=False,
                      all21_training_sets_checked=True, main_checkpoints_checked=models_checked,
                      diagnostic_checkpoints_checked=diagnostic_checked, release_files_checked=checked,
                      best_R=saved['R'], winner=saved['origin'], target_achieved=saved['R'] <= 104,
                      roles=roles, timing=timing, platform=platform.platform(),
                      raw_evidence=('evidence/formal/rounds' if args.archived else 'project/sessions/formal/rounds'))
    if args.archived:
        winner = read(archive / 'best-result.json')
        require(winner['R'] == saved['R'] and winner['origin'] == saved['origin']
                and winner['weights'] == saved['weights'], 'Archived winner export mismatch')
        require(read(archive / 'best-weights.json') == {'weights': [saved['weights']]},
                'Archived weight export mismatch')
        validation.update(evaluator_called=False, archived_selection_replayed=replay_selection, host_access_logs_included=False)
        atomic_json(ROOT / 'verification/ARCHIVE_VERIFICATION.json', validation)
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0
    atomic_json(ROOT / 'best-weights.json', dict(weights=[saved['weights']]))
    atomic_json(ROOT / 'best-result.json', dict(R=saved['R'], weights=saved['weights'],
                origin=saved['origin'], score_source='local_teacher_original_exe',
                official_executable_used=True, official_calls=20, evaluations=1000))
    with (ROOT / 'round-results.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['round','batch_min_R','cumulative_best_R','prediction_MAE','prediction_RMSE'])
        for row in summaries:
            writer.writerow([row[k] for k in ('round','batch_min_R','cumulative_best_R','prediction_MAE','prediction_RMSE')])
    hours = timing['session_wall_seconds']/3600
    strategy_label = source['strategy_label']
    report = [f'# CPU 独立寻优 · {strategy_label}', '',
              f"实验开始时间 {state['started_at']}", '',
              '## 结果与验证范围', '',
              f'本机从原始 2,048 行数据开始，使用老师提供的原版 `solver.exe` 完成 **20 次调用 × 每次 50 组 = 1,000 组新增评估**。',
              f'最终最低真实 R 为 **{saved["R"]:.12f}**，来源 `{saved["origin"]}`；目标 R≤104 **'+('达到' if saved['R'] <= 104 else '未达到')+'。',
              f'实验总历时约 **{hours:.2f} 小时**。结果是一次固定种子实验，不能据此宣称统计显著优势或全局最优。', '',
              '本次使用的服务器先验仅为第二栋楼peak、ramping、resilience支撑区域值得覆盖。未读取服务器参数、评分、模型或其他实验反馈；本次不是完全无服务器先验的盲搜。访问审计位于 project/access-audits/。', '',
              '## 问题与方法', '',
              '每个候选为三栋楼各七个非负权重，每楼权重和为 1，按楼宇顺序传入老师评分器。每楼七项顺序为 cost、carbon、peak、discomfort、degradation、ramping、resilience。目标是最小化评分器返回的标量 R；预测 R 与原版实测 R 分开保存。', '',
              f'前 6 轮沿用结构启发 1.0 版，第 7 轮起使用{strategy_label}。', '',
              '代理模型由三条互不共享训练参数的软树分支构成，每楼四棵深度 3 的软决策树，输出 32 维有符号隐特征。三楼特征相加后取绝对值并求均值，总参数量 3,692。隐特征只受标量 R 监督，不当作真实物理轨迹。', '',
              '每轮用原始数据与本实验已返回的反馈从头训练五种子模型；种子 2026–2030，epoch 分别 486、484、484、481、471。采用 float64、AdamW、学习率 0.001、weight decay 0.0001、batch 128，加权 MSE 与 0.02 倍成对排序损失。保留原版全量加权训练和自适应半径（倍率1.0）；每轮仅替换最多6个冲分候选，完整保留原版学习候选及成对关系。', '',
              '每轮 50 组包括冲分组和学习组：第 1–6 轮 30/20，第 7–14 轮 35/15，第 15–19 轮 40/10，第 20 轮 50/0，合计 710/290。所有候选由原版 EXE 评分后才加入训练；不以模型预测或插值充当真实标签。', '',
              '## 实验流程', '',
              '1. 校验原始 CSV、源码及老师 EXE 的 SHA-256，检查固定环境。',
              '2. 仅以原始 2,048 行训练 v00，生成第一批 50 个候选。',
              '3. 调用前先保存候选、预测、精确请求和预算记录；直接向原版 EXE 管道传入 JSON。',
              '4. 校验返回数量、有限值和输入顺序，保存 stdout、stderr、退出码及耗时，再加入本实验反馈。',
              '5. 累计反馈重训并进入下一轮，完成 20 次后使用全部 3,048 行训练 v20，不追加第 21 次评分。',
              '6. 独立核验 1,000 组精确配对、全部 21 版训练集、105 个主模型和诊断模型的哈希，再导出报告。', '',
              '## 本机逐轮结果', '',
              '| 轮次 | 当轮最低 R | 累计最低 R | 提交前预测 MAE |',
              '|---:|---:|---:|---:|']
    report.extend(f'| {r["round"]} | {r["batch_min_R"]:.9f} | {r["cumulative_best_R"]:.9f} | {r["prediction_MAE"]:.6f} |' for r in summaries)
    report += ['', '## 最优权重', '',
               '| 楼宇 | cost | carbon | peak | discomfort | degradation | ramping | resilience |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    report.extend('| 楼 '+str(b)+' | '+' | '.join(f'{v:.10g}' for v in w)+' |' for b,w in enumerate(saved['weights'],1))
    report += ['', '表格为便于阅读的显示精度，实际提交请使用保留完整浮点精度的 `best-weights.json`。', '',
               '## 环境、来源与限制', '',
               f'{platform.platform()}；Python 3.10.21、NumPy 1.26.4、PyTorch 2.6.0+cpu，两个训练进程、每个进程单线程。',
               f'源码基于 `Dookir/soft-tree-online-104` 提交 `{source["commit"]}`。老师 EXE 的 SHA-256 为 `{source["teacher_executable_sha256"]}`。',
               '训练内核沿用原版 CPU。候选入口为 acquisition.py，A 方案见 paired_selection.py。训练内核哈希见 verification/BASELINE_HASHES.json。',
               '本次只获取评分器标量 R，未独立取得逐小时物理轨迹，因此不宣称对原版评分内部计算作了独立数学复算。', '',
               '## 附件和复核', '',
               '`verification/LOCAL_VALIDATION.json`：本机证据核验；`round-results.csv`：逐轮数据；`best-result.json`：最优实测结果；`best-weights.json`：评分器输入格式；`project/sessions/formal/`：本次全部训练、候选和评分记录。']
    (ROOT / '实验报告.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    write_effects(rows)
    atomic_json(ROOT / 'verification/LOCAL_VALIDATION.json', validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


def write_effects(rows):
    details = []
    stats={};previous=min(r['R'] for r in records())
    for number in range(1,21):
        batch=[r for r in rows if r['round']==number]
        for r in batch:
            if not r['source'].startswith('specialized_'):continue
            st=stats.setdefault(r['source'],dict(count=0,best_R=1e99,new_records_vs_previous_round=0,parent_improvements=0))
            st['count']+=1;st['best_R']=min(st['best_R'],r['R'])
            st['new_records_vs_previous_round']+=int(r['R']<previous)
            m=r['acquisition']
            details.append(dict(id=r['id'], round=number, source=r['source'], R=r['R'],
                                improvement_vs_parent=m['parent_R']-r['R'] if 'parent_R' in m else None,
                                acquisition=m))
            if 'parent_R' in m:st['parent_improvements']+=int(r['R']<m['parent_R']-1e-12)
        previous=min([previous]+[r['R'] for r in batch])
    atomic_json(ROOT/'specialization-effects.json',stats)
    atomic_json(ROOT/'strategy-feedback.json', details)
    with (ROOT/'实验报告.md').open('a',encoding='utf-8') as f:
        f.write('\n## 特化名额真实收益\n\n')
        for source,st in stats.items():f.write(f"- {source}: {st['count']}组，最低R={st['best_R']:.12f}，优于上一轮累计最优{st['new_records_vs_previous_round']}组，优于自身父点{st['parent_improvements']}组。\n")


if __name__ == '__main__':
    raise SystemExit(main())
