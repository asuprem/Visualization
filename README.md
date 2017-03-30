# Visualization
There are three necessary python files: cifar_main.py, cifar_layers.py, and cifar_activations.py

cifar_main.py generates and trains the model. There are different versions - cifar_main0.py and cifar_main1.py. Each trains an entirely different topology on the CIFAR datasets. Modifying the training set is straightforward - replace the imports inside cifar_mainx.py with the necessry dataset (i.e. CIFAR-100). Running each on AWS gx2.2 large instance with 40 epochs takes 2 hours for cifar_main1.py and 2.5 hours for cifar_main2.py.

cifar_layers.py performs the gradient ascent for each layer. The model is built from the models stored by cifar_main.py, so running cifar_main.py is a prerequisite. There are a few relevant global variables i cifar_layers.py:

:_MODEL -> This lets the file know which model to use for gradient ascent. The models are named appropriately. You need to move all HDF% models inside  weights folder for this file to work (for now).

:_BASEDIM -> This is the dimension of the generated filter maximizers. Smaller umbers obviously run faster. Note that although the model is designed to work with 32x32 images for testing, during gradient ascent, any image size can be used, allowing for a more detailed look into the inner workings.

:_VISITERS -> This is the number of iterations for gradient ascent. We find 25-50 to be appropriate, and 25 to be reasonably fast (~5-6 mins on AWS gx2.2 large instance)

cifar_activations.py generates the list of CIFAR images that have the highest activation at each filter. I am working on generating a list of the top activations to be more useful.
