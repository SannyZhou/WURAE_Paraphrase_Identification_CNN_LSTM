# WURAE_Paraphrase_Identification_CNN_LSTM

Code & Data for our NLPCC 2018 paper [Paraphrase Identification Based on Weighted URAE, Unit Similarity and Context Correlation Feature](https://link.springer.com/chapter/10.1007/978-3-319-99501-4_4)

## Citation
Please cite as: <br>
		
	@inproceedings{zhou2018paraphrase,
	  title={Paraphrase identification based on weighted URAE, unit similarity and context correlation feature},
	  author={Zhou, Jie and Liu, Gongshen and Sun, Huanrong},
	  booktitle={CCF International Conference on Natural Language Processing and Chinese Computing},
	  pages={41--53},
	  year={2018},
	  organization={Springer}
	}
	

## Environment Setup:

	conda create -n env_name python=3.5
	pip install -r requirements.txt
	need to download GoogleNews-vectors-negative300.bin for the PI code test

## About code:

   1. Weighted_Unfolding_Recursive_AutoEncoders_Torch.py, TestRecusiveAutoEncoderTorch.py
		
    INFO: Weighted Unfolding Recursive Autoencoders model for phrase embedding and sentence embedding,
	train on a large scale of parse trees which is parsed from sentences by Stanford Parser
    RELATED DATA: ../data/sentence_msr_paraphrase_testparsed.txt, ../data/sentence_parsed.txt, torchweights/english_wurae.pkl, ../data/english_counter.pkl (english_counter.pkl refers to the frequency of words in the corpus, here only gives the sample result data of msrpc)
    IMPLEMENTATION: 
	(1) TEST THE PRETRAINED MODEL, calculate the phrase or sentence embedding: (-gpu 1 for using gpu, -parsed filename: filename should contain parse tree text, filename sample : ../data/sentence_msr_paraphrase_testparsed.txt, ../data/sentence_msr_paraphrase_trainparsed.txt)
		python TestRecursiveAutoEncoderTorch.py --model-weight torchweights/weightfilename -gpu 0 -parsed filename
	(2) TRAIN MODEL: 
		python Weighted_Unfolding_Recursive_AutoEncoders_Torch.py --batch-size 5000 -unfold 1 -gpu 1 -weighted 1 -model modelname
   
   2. msrp_sentence_level_parahraseDetection.py (picture of model : msrp_PI_model.png) (Run <1-Test> ,<4> before <2>)
  
	INFO: deep learning model of English sentence-level paraphrase identification, experiment on MSRPC
	RELATED DATA : ../data/msrpc_train_set.pkl, ../data/msrpc_test_set.pkl, ../data/msrpc_val_set.pkl, checkpoint_msrp_paraphrase_detection.hdf5(Because of file size, here only gives the origin data, so first run the code<3> and code<6> to get the data for training)
    
	IMPLEMENTATION: (output file of test: msrp_sentence_test_acc_f1)
	(1) TEST THE PRETRAINED MODEL:
		python msrp_sentence_level_parahraseDetection.py 
	(2) TRAIN MODEL: (default num_of_patience is 20)
		python msrp_sentence_level_parahraseDetection.py -test 0 --early-stopping num_of_patience
	
   3. chinese_article_level_parahraseDetection.py (picture of model : chinese_PI_model.png)
		
    INFO: deep learning model of Chinese article-level paraphrase identification, experiment on Chinese Sports & Entertainment NEWS Article paraphrase corpus
    RELATED DATA: ../data/chinese_train_set_with_others.pkl, ../data/chinese_test_set_with_others.pkl, ../data/chinese_val_set_with_others.pkl, checkpoint_chinese_paraphrase_detection.hdf5 (Because of file size, here only gives the sample data for training)
    IMPLEMENTATION: (output file of test: chinese_article_test_acc_f1)
	(1) TEST THE PRETRAINED MODEL:
		python chinese_article_level_parahraseDetection.py
	(2) TRAIN MODEL: (default num_of_patience is 15)
		python chinese_article_level_parahraseDetection.py -test 0 --early-stopping num_of_patience

   4. msrp_data_process.py run<1 test> before <4>
		
	INFO: preprocess data, english data for example, preprocess and split the origin train msrp to train, val set, preprocess origin test set
	RELATED DATA: ../data/msr_paraphrase_train.txt, torchweights/english_wurae_sentence_msr_paraphrase_trainparsed_nodeFeature.pickle, ../data/msr_paraphrase_test.txt, torchweights/english_wurae_sentence_msr_paraphrase_testparsed_nodeFeature.pickle, ../data/en.json, ../data/stopwords.dat
	IMPLEMENTATION: 
		python msrp_data_process.py

## About data:
  Those data should be included in the project. Because of the limitated file size, we would supply sample data.
  
	<1> GoogleNews-vectors-negative300.bin : english pre-trained word2vec
	<2> chinese_wordembedding_with_wiki_word2vec_format.bin : chinese word embedding trained from zh-wiki and sogou news
	<3> chinese_counter.pkl, english_counter.pkl: frequency of words in the corpus for weighted URAE
	<4> chinese_sentences_factor_parsed_tmp.txt, val_chinese_sentence_factor_parsed_tmp.txt : train and validation file for chinese WURAE training, parse tree of sentences from Sogo News and the train set of chinese news article paraphrase 
	<5> sentence_parsed.txt, sentence_msr_paraphrase_testparsed.txt: train and validation file for english WURAE training, parse trees of sentences from COCA NEWS, NEWS ON WEB, train set of msrpc
	<6> sentence_msr_paraphrase_trainparsed.txt, sentence_msr_paraphrase_testparsed.txt : parse tree file for origin msrpc train & test file
	<7> en.json, stopwords.dat : english and chinese stopwords file
	<8> chinese_train_set_with_others.pkl, chinese_val_set_with_others.pkl, chinese_test_set_with_others.pkl :(SAMPLE) preprocessed chinese train, validation, test file. The data could be loaded from the files by pickle, including coloumn of 'origin', 'repl', 'origin_sentence_embedding', 'repl_sentence_embedding', 'label', 'similarity', 'word_matrix_t', where 'origin' and 'repl' is the sentences list of articles, 'origin_sentence_embedding' and 'repl_sentence_embedding' is the list of sentence embedding of articles, 'similarity' is the sentence similarity matrix and 'word_matrix_t' is the transpose of 'similarity'.
	<9> msrpc_train_set.pkl, msrpc_val_set.pkl, msrpc_test_set.pkl : preprocessed msrpc train, validation, test file, including list of word embedding, word simiarlity matrix, node similairty matrix, bleu score, lcs, mini-edit, tf-idf, number feature,label, pair of sentences.
	<10> sample_chinese.pkl, sample_chinese_origin_trees_list.pkl, sample_chinese_repl_trees_list.pkl : for the test of chinese WURAE
	<11> msr_paraphrase_test.txt, msr_paraphrase_train.txt : origin msrp test and train set
	
   Data in code/torchweights:

	<1> english_wurae_sentence_msr_paraphrase_testparsed_nodeFeature.pickle, english_wurae_sentence_msr_paraphrase_trainparsed_nodeFeature.pickle :  phrase embedding list of msrp

