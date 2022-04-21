from usc_learning.envs.AlienGoGymEnv import AlienGoGymEnv
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np
import glob


# if __name__ == "__main__":
	# test out some functionalities
env = AlienGoGymEnv(isrender=True, control_mode="FORCE")

# fourcc = VideoWriter_fourcc(*'MP42')
# video = VideoWriter('./dummy.avi', fourcc, float(20), (480, 360))

# img_array = []
# img_array.append(env.render())
# video.write(env.render())
env.reset()
for i in range(2000):
    print(i)
    obs, reward, done, info = env.step(np.array([0,0]))
    # obs, reward, done, info = env.step(env.action_space.sample())
    # img_array.append(env.render())
    # if (i+1)%10 == 0:
    # video.write(env.render())

    if done:
        env.reset()

# out = cv2.VideoWriter('project.avi',fourcc = cv2.VideoWriter_fourcc(*'DIVX'), fps = int(1/0.005), frameSize = 20)
# video=cv2.VideoWriter('video.avi',-1,1,(width,height))

# for i in range(len(img_array)):
#     video.write(img_array[i])
# video.release()