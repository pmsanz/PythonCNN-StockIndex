# Portfolio
## Python - CNN StockIndex
### StockBuy/Sell Prediction Using Convolutional Neural Networks

Can an AI model predict the buying/selling order of a capital stock using the concept of image classification?
Insipired on the paper https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach

I create my own version of this to example to understand if this is possible.
While there is no real practical solution to this problem, it will at least satisfy our greedy curiosity as to whether it is possible to somehow automate our economies. We all know that there are promises of AI solutions on the capital market. But I promise you, this project will at least be interesting to watch.
A lot of technical details are ignored, so as not to bore or dizzy the viewer, any question or query that can be resolved will be welcome in the comments. I will first divide the project into parts. To then explain the operating process.


### Model:
- Convuntional Neural Network Image Classificator with 3 possible outputs.
- Adam optimizer
- Input dimension : 15 x 15 x 3 chanels
- EarlyStopping
- ReduceLROnPlateau

### Information:
- Historical data of the last 60 days.
- Real-time value of the BTC/USD index every 1 minute.
- More than ***+380*** technical indicators!
- istream.db : real data to predict
- tstream.db : test & train data

### Training
An algorithm for labeling of historical information is generated, using the min/max search with a specific time window. Later, the possibility of categorizing this information through other indicators is added to obtain a more ***"real"*** solution.

### Process:
1. From information retrieved by binance API on the value of the BTC / USDT indicator, more than 380 indicators are generated from close value and volume of the last time's period.
2. This information is saved in the DB, to be used later and generate a 3-channel RGB image, in my case I used an image of 15 x 15 pixels and filled with null values as ***"frame"*** of the image, for all those ***"spaces"*** empty of information. I could also have put other indicators to add more information. 
3. A deep CNN network is generated, with various filters and layers. With 3 possible outcomes. (this is simply trial and error, recovering information from other papers of similar projects to generate a model that is as functional as possible) This point has many technical details that are necessary to see in the code to be explained and it is up to the user to consult me for said information.
4. Training information
Using the information provided by a provider, something like 86,000 results of the values ​​of the last 60 days are tagged. Using a min/max search algorithm in a specified time window. Thus obtaining the states to classify ***(Buy - Sell - Hold)***
5. The CNN is trained with these values. (we obtain a value slightly greater than 70% according to the labeling algorithm used and the properties of the CNN)
6. Once the model has been trained, the value of the indicator is consumed in real time through the Binance API. The technical indicators for said value are generated and then the new "image" to be classified is generated. This image will determine the stock market action ***(Buy - Sell - Hold)***.
7. By means of different types of stop loss and controls, so as not to try to sell the action, when we have not yet made any purchase, for example. We create an application, which buys, sells or holds, according to the different types of indicators.

### Results:
Classify resulting image on whether it is necessary ***( buy / sell / hold )***

