# spam-ham-message-detection-using-machine-learng-Internship-

We all face the problem of spams in our inboxes. Let’s build a spam classifier program in python which can tell whether a given message is spam or not! We can do this by using a simple, yet powerful theorem from probability theory called Baye’s Theorem.


steps are:

1. Loading dependencies
2. Loading Data
3. Train-Test Split
To test our model we should split the data into train dataset and test dataset. We shall use the train dataset t0 train the model and then it will be tested on the test dataset. We shall use 75% of the dataset as train dataset and the rest as test dataset. Selection of this 75% of the data is uniformly random.
4. Training the model
We are going to implement two techniques: Bag of words and TF-IDF. I shall explain them one by one. Let us first start off with Bag of words.
Preprocessing: Before starting with training we must preprocess the messages. First of all, we shall make all the character lowercase. This is because ‘free’ and ‘FREE’ mean the same and we do not want to treat them as two different words.
5. Training the model
We are going to implement two techniques: Bag of words and TF-IDF. I shall explain them one by one. Let us first start off with Bag of words.
Preprocessing: Before starting with training we must preprocess the messages. First of all, we shall make all the character lowercase. This is because ‘free’ and ‘FREE’ mean the same and we do not want to treat them as two different words.
6. Classification
For classifying a given message, first we preprocess it. For each word w in the processed messaged we find a product of P(w|spam). If w does not exist in the train dataset we take TF(w) as 0 and find P(w|spam) using above formula. We multiply this product with P(spam) The resultant product is the P(spam|message). Similarly, we find P(ham|message). Whichever probability among these two is greater, the corresponding tag (spam or ham) is assigned to the input message. Note than we are not dividing by P(w) as given in the formula. This is because both the numbers will be divided by that and it would not affect the comparison between the two.

7. Final result

![image](https://user-images.githubusercontent.com/66793851/86527318-818ec280-bebb-11ea-9dc0-c6a9db5d9c8a.png)
![image](https://user-images.githubusercontent.com/66793851/86527318-818ec280-bebb-11ea-9dc0-c6a9db5d9c8a.png)
   
![ff](https://user-images.githubusercontent.com/66793851/86527462-856f1480-bebc-11ea-95fd-ed0a65670d17.png)
![q](https://user-images.githubusercontent.com/66793851/86527467-8d2eb900-bebc-11ea-9fa8-431c63e199e3.jpg)

  
