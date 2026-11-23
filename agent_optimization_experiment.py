"""Compare direction planning and choosing executable Sokoban pushes."""

import argparse
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from game_session import Run, GAME_RULES, DIRECTION_NAMES
from model_client import Model, ModelClient, ModelError, load_models, load_local_keys
from model_log import ModelCallLog
from sokoban import DIRECTIONS, LEVELS, move, reachable_box_squares, solve, won


ROOT = Path(__file__).resolve().parent

def offset(point, delta, n=1):
    return point[0] + n * delta[0], point[1] + n * delta[1]

def grid_stats(level, safe_squares=None):
    queue = deque([(level.initial, 0)])
    seen = {level.initial}
    expanded = 0
    while queue:
        board, depth = queue.popleft()
        expanded += 1
        if won(level, board):
            return {"distance": depth, "discovered": len(seen), "expanded": expanded}
        for direction in DIRECTIONS:
            successor = move(level, board, direction).board
            if successor in seen or (safe_squares is not None and not successor.boxes <= safe_squares):
                continue
            seen.add(successor)
            queue.append((successor, depth + 1))
    raise ValueError("No solution in reference levels")

def pushes(level, board, safe_squares):
    walks = {board.player: ()}
    queue = deque([board.player])
    while queue:
        position = queue.popleft()
        for action, delta in DIRECTIONS.items():
            nxt = offset(position, delta)
            if nxt in level.floor and nxt not in board.boxes and nxt not in walks:
                walks[nxt] = walks[position] + (action,)
                queue.append(nxt)
    result = {}
    for box in sorted(board.boxes):
        for action, delta in DIRECTIONS.items():
            stand = offset(box, delta, -1)
            dest = offset(box, delta)
            if stand in walks and dest in safe_squares and dest not in board.boxes:
                result[f'p{len(result)+1}'] = (box, dest, walks[stand]+(action,))
    return result

def offline():
    results={}
    for i,level in LEVELS.items():
        safe=reachable_box_squares(level)
        base, improved = grid_stats(level), grid_stats(level,safe)
        assert base['distance'] == improved['distance'] == len(solve(level))
        results[i]={'plain':base,'pruned':improved, 'initial_push_options':len(pushes(level,level.initial,safe))}
    all_runs=[]
    for f in sorted((ROOT / 'log').glob('benchmark_20261122-08*.jsonl')):
        for line in f.read_text(encoding='utf-8').splitlines():
            trial=json.loads(line)
            if trial.get('type')!='trial': continue
            level=LEVELS[trial['level']]
            board=level.initial
            safe=reachable_box_squares(level)
            disallowed_at=None
            dead_at=None
            for i,action in enumerate(trial['actions'],1):
                result=move(level,board,action['direction'])
                assert result.outcome==action['outcome']
                board=result.board
                if disallowed_at is None and not board.boxes <= safe:
                    disallowed_at=i
                if dead_at is None and not won(level,board) and solve(level,board) is None:
                    dead_at=i
            assert dead_at==trial['deadlock_at'], (trial['run_id'],dead_at,trial['deadlock_at'])
            all_runs.append((dead_at,disallowed_at,trial['status']))
    summary={'solver':results, 'replayed':len(all_runs),
             'deadlocks':sum(d is not None for d,_,_ in all_runs),
             'pruned_at_first_deadlock':sum(d is not None and d==s for d,s,_ in all_runs),
             'original_wins':sum(status=='solved' for _,_,status in all_runs)}
    print(json.dumps(summary,ensure_ascii=False),flush=True)

def baseline_trial(level, model, client, call_limit):
    run=Run(level,"model",model,observation='text',strategy='plan_all',horizon=8)
    trace=[]
    first_deadlock=None
    steps=0
    while run.status=='running' and (run.pending or run.calls<call_limit):
        result=run.step({'revision':run.revision},client)
        if result['plan']:
            api=result['plan'].get('api',{})
            trace.append((api.get('call_id'),api.get('status'),result['plan']['error']))
        if result['action']:
            steps+=1
            if first_deadlock is None and solve(level,run.board) is None:
                first_deadlock=steps
    return {'won':won(level,run.board),'calls':run.calls,'moves':run.moves,'blocked':run.blocked,
            'deadlock':first_deadlock,'wait_ms':run.wait_ms,'trace':trace,
            'input':run.input_tokens,'output':run.output_tokens}

def candidate_trial(level, model, client, call_limit):
    board=level.initial
    safe=reachable_box_squares(level)
    run_id=uuid4().hex
    call_num=moves=wait_ms=inp=out=0
    dead_at=None
    trace=[]
    history=[]
    actions=[]
    while not won(level,board) and call_num<call_limit and moves<40:
        options=pushes(level,board,safe)
        if not options: break
        options_text=[]
        for label,(box,dest,path) in options.items():
            name=DIRECTION_NAMES[path[-1]]
            cell=lambda pt: f'({pt[0]+1},{pt[1]+1})'
            flags = "，推入目标" if dest in level.goals else "，离开目标" if box in level.goals else ""
            options_text.append(f'{label}：走 {len(path)-1} 格后向{name}推箱 {cell(box)} 到 {cell(dest)}{flags}')
        observation = ('字符棋盘：# 墙，空格 地板，@ 玩家，$ 箱子，. 目标，+ 玩家站目标，* 箱子在目标。'
            '\n坐标使用（行，列），从左上角 (1,1) 开始。\n'+'\n'.join(level.text(board))+
            '\n最近执行：'+'；'.join(history[-3:])+'\n可执行候选：\n'+'\n'.join(options_text))
        instructions = GAME_RULES + (
            '程序会走到指定站位并推箱一次。根据当前棋盘，选出最有助于通关的一个候选，避免撤销刚走过的进展。'
            '可先横向调整箱子以便之后对准目标。只输出候选 ID，例如 p1。'
        )
        call_num+=1
        context={"run_id":run_id,"level":level.id,"plan":call_num,
                 "observation":"text","strategy":"push_candidate_prototype","horizon":1}
        try:
            c=client.complete(model,instructions,observation,None,context=context)
        except ModelError as e:
            wait_ms+=e.elapsed_ms
            trace.append((e.trace.call_id if e.trace else None,e.trace.status if e.trace else None,str(e)))
            continue
        wait_ms+=c.elapsed_ms
        inp+=c.input_tokens or 0
        out+=c.output_tokens or 0
        choice = c.text.strip().strip(chr(96)+'"').strip().removesuffix('.').strip()
        if choice not in options:
            trace.append((c.trace.call_id,c.trace.status,'invalid candidate ID'))
            continue
        trace.append((c.trace.call_id,c.trace.status,''))
        index=list(options).index(choice)
        _,_,path=options[choice]
        if moves+len(path)>40: break
        history.append(options_text[index])
        for action in path:
            result=move(level,board,action)
            assert result.outcome!='blocked'
            board=result.board
            moves+=1
            actions.append(action)
        if dead_at is None and not won(level,board) and solve(level,board) is None:
            dead_at=moves
    return {'won':won(level,board),'calls':call_num,'moves':moves,'blocked':0,
            'deadlock':dead_at,'wait_ms':wait_ms,'trace':trace,'actions':actions,'input':inp,'output':out}

def live():
    load_local_keys(ROOT / '.env')
    models=load_models(ROOT / 'models.json')
    catalog=[
        models['openrouter:google/gemini-2.5-flash'],
        models['openrouter:openai/gpt-4.1-mini'],
        Model('openrouter','openai/gpt-4o-mini','GPT-4o mini',True),
        Model('openrouter','openai/gpt-4o','GPT-4o',True),
    ]
    if not all(model.available for model in catalog):
        raise SystemExit("请先在项目 .env 或环境变量里配置 OPENROUTER_API_KEY。")
    log=ModelCallLog(ROOT / 'log')
    client=ModelClient(call_log=log)
    print('log',log.path.name,flush=True)
    def unit(pair):
        model,level_id = pair
        base=baseline_trial(LEVELS[level_id],model,client,4)
        option=candidate_trial(LEVELS[level_id],model,client,4)
        result={'model':model.label,'level':level_id,'base':base,'proposal':option}
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(unit,[(model,level_id) for model in catalog for level_id in ('2','3')]))
    output={'log':log.path.name,'results':results}
    print(json.dumps({'hits_base':sum(row['base']['won'] for row in results),'hits_proposal':sum(row['proposal']['won'] for row in results)},ensure_ascii=False),flush=True)
    summary_path = log.path.with_name(log.path.name.replace('model_calls_', 'push_candidates_', 1)).with_suffix('.json')
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('summary', summary_path.name)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="run up to 64 OpenRouter requests")
    if parser.parse_args().live:
        live()
    else:
        offline()


if __name__ == "__main__":
    main()
