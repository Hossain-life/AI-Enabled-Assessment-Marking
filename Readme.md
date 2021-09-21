# Problem Statement

I developed a learning system to perform an intelligent non-STEM assessment marking such as
marking an Essay (a short piece of writing on a particular subject such as English, History, and Geography).


# Data description

There are 4 columns in the data which are essay_id, essay, essay_set and score. There are 8 types of essay questions, and for 8 questions, there are 8 types of scores. It means that
for 8 essay questions, the questions are marked differently. 

# Data Preprocessing

As there are 8 essay questions, I made 8 set of columns for 8 essay_set questions. Then, I filtered the data rows based on people who attempted first essay questions. Subsequently, 
I used mapping to create a final grade column which contains 8 distinct categorical values, and later on, I computed the count values for all the classes in the label column. I saw that
all the class values are imbalanced. In order to solve the imbalanced problem, I converted my classifier into binary problem, and then upsampled the minority classes.


# Fine tune bert for text classification

To implement fine tune bert for text classification, I installed Huggingface’s transformers library, which imports a wide range of transformer-based pre-trained models. After that, I I
splitted the dataset into three sets – train, validation, and test, and I imported BERT-base model that has 110 million parameters. Since the messages (essay) in the dataset are of varying length, I used padding to make all the messages have the same length. 
I used the maximum sequence length to pad the messages, and in order to do so, I looked at the distribution of the sequence lengths in the train set to find the right padding length. Then, I tokenized and encoded sequences in the training set, validation set
and test set, and I converted lists to tensors. Next, I created dataloaders for both train,and validation set, which passed batches of train data and validation data as input to the model during the training phase. 
Later, I frooze all the layers of the model before fine-tuning it which prevented updating of model weights during fine-tuning, and consequently, I declared Bert architecture where I
used  AdamW as our optimizer- an improved version of the Adam optimizer. I computed the class weights for the target column, and then passed the weights to the loss function. Next, I converted list of class weights to a tensor, and I pushed to GPU. 
Consequently, I defined the loss function and the number of training epochs, and I also defined a couple of functions to train (fine-tune) and evaluate the model, respectively. 
I also used evaluate function to evaluate the model which uses the validation set data, and then, I started fine-tuning of the model. To make prediction, I loaded the best model weights which were saved during the training process.
Once the weights are loaded, I used the fine-tuned model to make predictions on the test set, and then, I checked out the model’s performance.



I again have to do the repeatitive tasks for the other essay questions. I only showed how it can be dealt with the scores of first set essay questions.
