This folder contains sample config files for two popular OpenAI gym environments, and two versions of spaceinvaders.

If you would like to train the game in greyscale, make sure the first entry in the input list is 'GREYSCALE'
The next input should be 'SCREENINPUT' which will allow the code to know that the next input will contain the amount of input nodes
After 'SCREENINPUT', enter 'Number of input nodes'. This number should be all the dimensions of the OBSERVATION_SPACE of the input game multiplied together, so in the case of spaceinvadersRAM, it would be 128, and in the case of spaceinvadersRGB-greyscale, it would be 33600 (210*160, if you wanted to train on the full colors, it would be 100800, 210*160*3)
