### Deep Learning (MTSU)  Project Demo

In the first place, clone the repository via `https://github.com/ttorku/Fall-2021-DL-Demo.git` on the command line or by downloading the zip file from the root page of this repository. You will have immediate access to all the files needed to run this project. 

The JupyterLab notebook named  ` Data_Driven_Simulation.ipynb` is used to run most of the output contained in the paper: `https://github.com/ttorku/Fall-2021-DL-Demo/blob/main/Deep_Learning_Research_Paper.pdf`. The auxillary scripts for running the notebook include: `EINN_model` for parameter estimation; `rnn_models` for RNN such as LSTM, BiLSTM and GRU as well as ResNet-GRU, ResNet-LSTM, ResNet-BiLSTM ; and `cross_val` for cross validation

The neccessary packages to be installed include the following:
- Pandas
- Numpy
- Matplotlib
- Tensorflow >1.5
- Pytorch
- Sklearn
- Time
- Dates
- pyDOE 
- scipy 
- The auxillary scripts are also imported.

A piece of text is included at the top of the notebook explaining what the block of code is doing. 

To execute a cell, click on it to select it and Shift+Enter.

The first few block of codes show output folders for Results and Models. The Results will contain results from graphs and Modles will have the resnet pre-trained model. 

The next block of codes describe the results from all the RNN models-GRU, LSTM and BiLSTM; Hybrid Approaches- ResNet-GRU, ResNet-LSTM, ResNet-BiLSTM.

The Error Metrics such as **RMSE, MAPE and EV** are obtained from these RNN and Hybrid Approaches.

A plot is shown to describe the results. 

The next block of codes implements the cross validation followed by the EINN implementation of vaccine model without and with vaccine given efficacy of $94\%. 




