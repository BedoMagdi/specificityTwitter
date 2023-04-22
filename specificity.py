import argparse
import pickle
import sys
import os, time
from createFeatures import *

MODELFILE = "./model/best_model.pkl"


# Load the model with best performance.

def getFeatures(fin):
    ## main function to run specifictTwitter parser and return predictions
    ## sentlist should be a list of sentence strings, tokenized;
    print("Start initialize word_embedding ...")
    embeddings = features.init_embeding()
    print("finished word_embedding ...")
    a = ModelNewText(embeddings=embeddings)

    ## When use our data.
    # a.loadFromCSV(fin)

    #a.loadFromFile(fin)
    a.loadFromTSV(fin)
    a.transLexical()
    a.transEmbedding()
    a.transEmotionFeature()
    a.transDeixisFeature()
    a.transform_features()


def predict(model=MODELFILE):
    with open(model, 'rb') as file:
        pickle_model = pickle.load(file)
        print("successfully laod")
    f = pd.read_csv("./output/test.csv", sep="\t")
    feature = f.iloc[:, 3:]
    output = pickle_model.predict(feature)
    return output


def writeSpecificity(preds, outf):
    with open(outf, "w") as f:
        for x in preds:
            f.write("%f\n" % x)
        f.close()
    print("output to " + outf + " done")
    clean()


def run(identifier, sentlist):
    ## main function to run the parser and return predictions
    ## sentlist should be a list of sentence strings, tokenized;
    ## identifier is a string serving as the header of this sentlst
    print("Start initialize word_embedding ...")
    embeddings = features.init_embeding()
    print("finished word_embedding ...")
    a = ModelNewText(embeddings=embeddings)
    a.loadFromFile(fin)
    a.transLexical()
    a.transEmbedding()
    a.transEmotionFeature()
    a.transform_features()
    return predict(model=MODELFILE)


def clean():
    # clean the intermediate files.
    os.remove("NE_Concrete_Emo.csv")
    os.remove("sample-tagged.txt")
    os.remove("USEFUL_TAG.csv")


def test():
    import nltk
    from nltk.util import ngrams

    from sklearn.feature_extraction.text import CountVectorizer

    # Example tweet text
    tweet = 'Hoping there is no #Caddyshack moment in the #Olympics2016 pool ! #BabyRuth #DOODIE'

    # Define the n-gram range
    ngram_range = (2, 3)  # Example: representing bigrams and trigrams

    # Initialize CountVectorizer with desired n-gram range
    vectorizer = CountVectorizer(ngram_range=ngram_range)

    # Fit and transform the tweet text to obtain n-gram features
    ngram_features = vectorizer.fit_transform([tweet])

    # Convert the sparse matrix to an array
    ngram_features_array = ngram_features.toarray()

    # Print the n-gram features
    print(ngram_features_array)

if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--inputfile",
                           help="input raw text file, one sentence per line, tokenized",
                           required=True)
    argparser.add_argument("--outputfile",
                           help="output file to save the specificity scores",
                           required=True)
    sys.stderr.write(
        "Predictor: please make sure that your input sentences are WORD-TOKENIZED for better prediction.\n")
    args = argparser.parse_args()
    getFeatures(args.inputfile)
    clean()

    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken: {:.2f} seconds".format(time_taken))

    # test()
    # preds = predict(model=MODELFILE)
    # writeSpecificity(preds, args.outputfile)
