This folder contains sample config files for two popular OpenAI gym environments, and three versions of spaceinvaders.

The input section of the config files now have extra options

If you type in "SCREENINPUT" for the first item in the input section, the following input should be a number equal to the total amount of inputs
EX 
input   = ["SCREENINPUT", 33600]

In the wrappers section, you can type in wrappers that should be applied to the environment
EX
wrappers    = ["gymnasium.wrappers.GrayScaleObservation"]
Make sure that your values in the input and output sections of the config match