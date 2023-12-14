from game import OthelloEnv, get_legal_moves, make_move,Human
from tqdm import tqdm
import asyncio
import time
from reversi_agent import ReversiAgent


BLACK_PLAYER = 1
WHITE_PLAYER = -1


async def timer(limit):
    """Create a progress bar for timer."""
    for i in tqdm(range(limit*10), desc="Time Limit: ", leave=False):
        await asyncio.sleep(1/10)


async def main(
        black_agent: ReversiAgent,
        white_agent: Human,
        timelimit: int=2):
    env = OthelloEnv(render=True, verbose=False)
    board, __, over, info = env.reset()
    current_player = info['player_to_take_next_move']
    winner = info['winner']


    for __ in range(2000):
        active_player = black_agent
        player_name = 'BLACK'
        if current_player == WHITE_PLAYER:
            active_player = white_agent
            player_name = 'WHITE'
        print('===================================================')
        print(f"{player_name}'s turn...")
        valid_actions = get_legal_moves(board, current_player)
        if len(valid_actions) > 0:
            try:
                start_time = time.time()
                if(current_player == BLACK_PLAYER):
                    agent_task = asyncio.create_task(
                        active_player.move(board, valid_actions))
                else:
                    agent_task = asyncio.create_task(
                        active_player.human_player_wrapper(board)
                    )
                time_task = asyncio.create_task(timer(timelimit))
                done, pending = await asyncio.wait(
                    {time_task, agent_task},
                    timeout=timelimit,
                    return_when=asyncio.FIRST_COMPLETED)
                time_task.cancel()
                agent_task.cancel()

            except asyncio.TimeoutError:
                d = time.time() - start_time - timelimit
                print(f'Timeout! Overtime: {d:.2}')
                break
            if current_player == BLACK_PLAYER:
                move = active_player.best_move
            else:
                move = active_player._move
            print(f'{player_name} made {move}')
        else:
            move = None
        if move == (-1, 0):
            move = None
        board, __, over, info = env.step(move)
        current_player = info['player_to_take_next_move']
        winner = info['winner']
        # print(current_player, over)

        if over:
            print('===================================================')
            if winner == BLACK_PLAYER:
                print('BLACK PLAYER WINS!')
            elif winner == WHITE_PLAYER:
                print('WHITE PLAYER WINS!')
            else:
                print('DRAW!')
            print('===================================================')
            break

if __name__ == "__main__":
    from reversi_agent import RandomAgent, MinimaxAgent,AlphaBetaAgent
    from game import Human
    black = AlphaBetaAgent(BLACK_PLAYER)
    white = Human(WHITE_PLAYER)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(black, white, 10))
    input('Press Enter to close.')