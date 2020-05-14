# tensorflow_gail_ppo

This project implemented gail with tensorflow which has been test by gym env "Pendulum-v0".The expert data is collected by model trained by PPO,which is saved in expert_data.

You can run "train.py" to train your own model.If you choose args.algo=="ppo",you can use ppo to train your model,and if args.algo=="gail" means you can train your model by gail.

The result in Pendulum-v0 env,The model is not trained to be very convergent, but you can still see that gail imitates expert data very well.

ppo mean rewards:

![image](https://github.com/zstbackcourt/tensorflow_gail_ppo/blob/master/pic/ppo_reward.png)

gail dloss:

![image](https://github.com/zstbackcourt/tensorflow_gail_ppo/blob/master/pic/gail_dloss.png)

gail reward(env reward):

![image](https://github.com/zstbackcourt/tensorflow_gail_ppo/blob/master/pic/gail_reward.png)
