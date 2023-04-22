@echo off
set "cur_path=%cd%"
cd %cur_path%\twitie-tagger\stanford-postagger
java -mx2000m -classpath stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -model models\gate-EN-twitter.model -textFile %cur_path%\input.txt > %cur_path%\sample-tagged.txt
cd %cur_path%