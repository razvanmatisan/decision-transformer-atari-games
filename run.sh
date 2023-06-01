# Decision Transformer (DT)

for seed in 123
do
    python run_dt_atari.py --seed $seed --context_length 50 --epochs 10 --model_type 'reward_conditioned' --num_steps 100000 --num_buffers 50 --game 'MontezumaRevenge' --batch_size 64
done


# python run_dt_atari.py --seed 123 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 1000000 --num_buffers 50 --game 'PrivateEye' --batch_size 64
# python run_dt_atari.py --seed 123 --context_length 30 --epochs 10 --model_type 'reward_conditioned' --num_steps 1000000 --game 'MontezumaRevenge' --batch_size 64
# for seed in 123
# do
    # python run_dt_atari.py --seed $seed --context_length 20 --epochs 5 --model_type 'reward_conditioned' --num_steps 1000000 --num_buffers 50 --game 'MontezumaRevenge' --batch_size 64
# done

# for seed in 123 231 312
# do
#     python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
# done

# for seed in 123 231 312
# do
#     python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
# done

# for seed in 123 231 312
# do
#     python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
# done

# Behavior Cloning (BC)
# for seed in 123
# do
#     python run_dt_atari.py --seed $seed --context_length 20 --epochs 5 --model_type 'naive' --num_steps 5000 --num_buffers 50 --game 'MontezumaRevenge' --batch_size 64
# done

# for seed in 123 231 312
# do
#     python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
# done

# for seed in 123 231 312
# do
#     python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
# done

# for seed in 123 231 312
# do
#     python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
# done