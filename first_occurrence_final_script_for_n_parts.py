import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import re
import numpy as np

#dictionary of contractions or slangs to convert back to original form
contractions_slangs = {
"aren't": "are not",
"can't": "can not",
"can't've": "can not have",
"cannot": "can not",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
"gonna": "going to",
"wanna": "want to"
}

#some common non-stopwords that are irrelevant to engineering design
irrelevant_words = [
    "us",
    "thought",
    "said",
    "think",
    "say",
    "thank",
    "went",
    "yeah",
    "yea",
    "yes",
    "awesome",
    "excellent",
    "great",
    "like",
    "ok",
    "okay",
    "so",
    "cool",
    "sure",
    "thing",
    "go",
    "went",
    "get",
    "also",
    "got",
    "would",
    "could",
    "tri",
    "guy",
    "dude",
    "soon",
    "stuff",
    "huh",
    "oh",
    "mm",
    "uh",
    "um"
]

#1. read in data from textfile
#1.1 establish pandas dataframe to contain data (columns are timestamp, speaker, sentence)

number_of_parts = int(input("Number of parts: "))
conversation = pd.DataFrame()

for i in range(number_of_parts):

	#establish delimiter
	raw_data = open(input("transcript file part {} path and name (.txt): ".format(i+1))).read()
	raw_data = raw_data.replace("  ", "\n")
	raw_data = raw_data.replace("\n\n", "\n")

	#organize into different lists (fortunately the data is organzied sequentially in the textfile)
	parsed_data = raw_data.split("\n")
	parsed_data = parsed_data[:-2] #get rid of otter statement at the end. Comment out if no otter statement

	speakers = parsed_data[0::3]
	timestamps = parsed_data[1::3]
	sentences = parsed_data[2::3]

	temp_df = pd.DataFrame({
	    'Speaker': speakers,
	    'TimeStamp': timestamps,
	    'Sentences': sentences 
	})

	conversation = conversation.append(temp_df, ignore_index = True, sort = False)

#2 clean up sentences
def process_text(text):

	#2.1 convert all letters to lower case
	try:
		text = str(text).lower()
	except AttributeError:
		print("Text causing error: ", text)	

	#2.2 break up contractions and correct slangs
	for word in contractions_slangs:
		if word in text:
			text = text.replace(word, contractions_slangs[word])

	#2.3 remove all punctuation (maybe except for - ?)
	text = text.replace("/", " ") #/ is often used for joining two words in the transcripts
	text = text.replace("'s", "")
	text = "".join([char for char in text if (char == "-") or (char not in string.punctuation)])
	
	#2.4 tokenize
	tokenized_text = re.split('\s+', text)

	#2.5 get rid of numbers and empty strings
	tokenized_text = [word for word in tokenized_text if (bool(re.match("[0-9]+", word)) == False) and (word != "...") and (word != "")]

	#2.6 remove stop words and other irrelevant words
	stopwords = nltk.corpus.stopwords.words('english')
	tokenized_text = [word for word in tokenized_text if (word not in stopwords) and (word not in irrelevant_words)]

	#2.7 lemmatize or stem
	#wn = nltk.WordNetLemmatizer()
	#tokenized_text = [wn.lemmatize(word) for word in tokenized_text]
	ps = nltk.PorterStemmer()
	tokenized_text = [ps.stem(word) for word in tokenized_text]

	#2.8 second round of clean up after lemmatization or stemming
	tokenized_text = [word for word in tokenized_text if (word not in stopwords) and (word not in irrelevant_words)]

	return tokenized_text

conversation["processed_sentences"] = conversation["Sentences"].apply(lambda x: process_text(x))

#3. divide up conversation into equal time segments
#3.1 convert the timestamps to "seconds elapsed" form for easier arithmetic and comparison
timestamps = [timestamp.split(":") for timestamp in conversation["TimeStamp"]]
timestamps = [[int(num) for num in timestamp] for timestamp in timestamps]
timestamps = [timestamp[::-1] for timestamp in timestamps]

converted_timestamps_initial = []
new_time = 0 #new time is number of seconds elapsed

for timestamp in timestamps:
    for i in range(len(timestamp)):
        new_time += pow(60, i)*timestamp[i]
        
    converted_timestamps_initial.append(new_time)
    new_time = 0

conversation["Converted_TimeStamp_Initial"] = converted_timestamps_initial

#join the n separate timestamps together
converted_timestamps = []
prev = 0
begin_index = 0

while len(converted_timestamps) != len(converted_timestamps_initial):

	new_ref = prev

	for i in range(begin_index, len(converted_timestamps_initial)):
		if converted_timestamps_initial[i] + new_ref >= prev:
			converted_timestamps.append(converted_timestamps_initial[i] + new_ref)
			prev = converted_timestamps[-1]
		else: #the current timestamp is smaller than the previous => we've hit the beginning of second part
			begin_index = i
			print("Joining timestamps, encountered discontinuity, resetting reference point. Current timestamp: {}".format(converted_timestamps_initial[i]))
			break

conversation["Converted_TimeStamp"] = converted_timestamps

#3.2 find out total duration of session
total_duration = conversation["Converted_TimeStamp"][len(conversation["Converted_TimeStamp"])-1] - conversation["Converted_TimeStamp"][0]
#round up to nearest minute?

#3.3 determine number of segments wanted
num_segements = 10

#3.4 calculate duration of each segment. Segments shall include left edge, exclude right edge
duration_per_segment = total_duration/num_segements

#4. form dictionary to hold the words spoken in each time segment by each speaker
'''
{
	speaker1: {
		timeseg1: [], #list of words
		timeseg2:[],
		...
	},

	speaker2: {
		timeseg1: [],
		timeseg2:[],
		...
	}
}

'''
#identify speakers
speakers = list(dict.fromkeys(conversation["Speaker"]))

#identify time segments
time_segs = []
start_time = conversation["Converted_TimeStamp"][0]
for i in range(num_segements):
	seg = [round(start_time+i*duration_per_segment, 2), round(start_time+(i+1)*duration_per_segment, 2)]
	time_segs.append(seg)

time_segs_strings = [str(seg) for seg in time_segs] #use as dict keys for now

#make dictionary with the speakers as keys
words_spoken_per_seg = dict.fromkeys(speakers)

#populate the value of each key with dictionaries with timesegment keys and initialize values as empty lists
for speaker in words_spoken_per_seg:
    words_spoken_per_seg[speaker] = dict.fromkeys(time_segs_strings)

    for seg in words_spoken_per_seg[speaker]:
    	words_spoken_per_seg[speaker][seg] = []


#concatenate cleaned word lists as the values for each timeseg key
for ind in conversation.index:
	for seg in time_segs:
		if (conversation["Converted_TimeStamp"][ind] >= seg[0]) and (conversation["Converted_TimeStamp"][ind] <= seg[1]):
			words_spoken_per_seg[conversation["Speaker"][ind]][str(seg)] +=  conversation["processed_sentences"][ind]
			break #prevent double counting


#5. count number of first occurences for each speaker in each timeseg, store in new dict
'''
{
	speaker1: {
		timeseg1: <num of first occurences>,
		timeseg2: ,
		...
	},

	speaker2: {
		timeseg1: ,
		timeseg2: ,
		...
	}
}
'''
#make dictionary with the speakers as keys
first_occurence_per_seg = dict.fromkeys(speakers + ["Combined"])

#populate the value of each key with dictionaries with timesegment keys and initialize values as 0
for speaker in first_occurence_per_seg:
    first_occurence_per_seg[speaker] = dict.fromkeys(time_segs_strings)

    for seg in first_occurence_per_seg[speaker]:
    	first_occurence_per_seg[speaker][seg] = 0


#list to contain the words that have been counted for first occurrnce
words_spoken = []

#loop through each list at each time seg
for speaker in words_spoken_per_seg:
	for seg in words_spoken_per_seg[speaker]:
		for word in words_spoken_per_seg[speaker][seg]:
			#if the encountered word is not in the unique words list, append it to the list and increment first occurrence counter for both the speaker and combined
			if word not in words_spoken:
				words_spoken.append(word)
				first_occurence_per_seg[speaker][seg] += 1
				first_occurence_per_seg["Combined"][seg] += 1
			#if the encountered word is in the unique words list, no action required

#convert to cumulative values
cumulated = 0
cumulative_first_occurence_per_seg = dict.fromkeys(speakers + ["Combined"])

#initialize values to 0
for speaker in cumulative_first_occurence_per_seg:
	cumulative_first_occurence_per_seg[speaker] = dict.fromkeys(time_segs_strings)

	for seg in cumulative_first_occurence_per_seg[speaker]:
		cumulative_first_occurence_per_seg[speaker][seg] = 0

for speaker in first_occurence_per_seg:
	for seg in first_occurence_per_seg[speaker]:
		cumulated += first_occurence_per_seg[speaker][seg]
		cumulative_first_occurence_per_seg[speaker][seg] += cumulated

	cumulated = 0


#6. plot bar graphs

#contructing horizontal axis values. use the xx:xx time stamp format on the x axis
def convert_back_timestamps(timestamp):
	temp_list = []
	val = timestamp

	for i in range(3):
		temp_list.append(round(val%60))
		val = int(val/60)

	temp_list = [str(num) for num in temp_list if num != 0]
	for x in range(len(temp_list)):
		if len(temp_list[x]) == 1:
			temp_list[x] = "0" + temp_list[x]

	temp_list = temp_list[::-1]
	
	return ":".join(temp_list)


horiz_axis_cumulative = [convert_back_timestamps(seg[1]) for seg in time_segs]
horiz_axis = [[convert_back_timestamps(timestamp) for timestamp in seg] for seg in time_segs]

#convert the elements to string so they can be displayed on bar graph
horiz_axis_cumulative = [str(seg) for seg in horiz_axis_cumulative]
horiz_axis = [str(seg) for seg in horiz_axis]

#contructing vertical axis values, use dictionary to organize the values for each speaker
vert_axis = dict.fromkeys(speakers + ["Combined"])
vert_axis_cumulative = dict.fromkeys(speakers + ["Combined"])

for speaker in vert_axis:
	vert_axis[speaker] = [first_occurence_per_seg[speaker][seg] for seg in first_occurence_per_seg[speaker]]

for speaker in vert_axis_cumulative:
	vert_axis_cumulative[speaker] = [cumulative_first_occurence_per_seg[speaker][seg] for seg in cumulative_first_occurence_per_seg[speaker]]

#6.2 bar graphs for combined number of first occurences (should use line graph instead. This was initial rough work)
def plot_bar_graph(vert_axis, horiz_axis, speakers, cumulative): #dictionary, list, dictionary, string
	
	width = 0.2
	i = 0
	index0 = np.arange(num_segements) #temp. numerical values 0-9 for horizontal axis
	indices = dict.fromkeys(speakers)

	for speaker in indices:
		indices[speaker] = [num + width*i for num in index0] #incremented horizontal locations for each bar
		i += 1

	for speaker in speakers:
		plt.bar(indices[speaker], vert_axis[speaker], width, label=speaker)

	plt.xticks(index0 + width*(i-1)/2, horiz_axis, fontsize=10, rotation=30)

	plt.xlabel('Time Segments', fontsize=12)
	plt.ylabel('Num. of First Occurences', fontsize=12)
	plt.title('{}First Occurences Count per Time Segment for '.format(cumulative) + speaker)
	plt.legend()

	plt.show()

#line graph for same data
def plot_line_graph(vert_axis, horiz_axis, speakers):
    index = np.arange(num_segements)
    
    df = pd.DataFrame({
        "horiz_values": index
    })
    
    for speaker in vert_axis:
        df[speaker] = vert_axis[speaker]
    
    for speaker in speakers:
        plt.plot("horiz_values", speaker, data = df, marker='.', label = speaker)
        
    plt.xticks(index, horiz_axis, fontsize=10, rotation=30)
    plt.xlabel('Time Segments', fontsize=12)
    plt.ylabel('Num. of First Occurences', fontsize=12)
    plt.title('Cumulative First Occurences Count per Time Segment')
    plt.legend()

    plt.show()

#speakers = [speakers[0]]+[speakers[1]]+[speakers[3]]+[speakers[2]] #cosmetic rearrange in eaiser to read order for generating graph (practitioner)
#speakers = [speakers[1]]+[speakers[0]]+[speakers[2]] #cosmetic rearrange in eaiser to read order for generating graph (academic) 

#plot_bar_graph(vert_axis_cumulative, horiz_axis_cumulative, speakers+["Combined"], "Cumulative")

plot_line_graph(vert_axis_cumulative, horiz_axis_cumulative, speakers+["Combined"])


#7. output raw data to excel
#make a list of all unique words spoken
words_spoken2 = []
for ind in conversation.index:
    for word in conversation["processed_sentences"][ind]:
        if word not in words_spoken2:
            words_spoken2.append(word)

#construct a dictionary whose keys are all the unique words spoken, and the corresponding value is a list that will hold the time
#when that word was first spoken and the person who spoke it
combined_first_occurence = dict.fromkeys(words_spoken2)
for word in combined_first_occurence:
    combined_first_occurence[word] = []

for word in combined_first_occurence:
    for ind in conversation.index:
        if word in conversation["processed_sentences"][ind]:
            combined_first_occurence[word].append(conversation["Speaker"][ind])
            combined_first_occurence[word].append(conversation["Converted_TimeStamp"][ind])
            break

#extract the speakers and times into their own separate lists
combined_speakers_list = [combined_first_occurence[word][0] for word in combined_first_occurence]
combined_times_list = [combined_first_occurence[word][1] for word in combined_first_occurence]

#make a DataFrame out of the three parallel lists - work spoken, time it first occurred, person who said it
combined_df = pd.DataFrame({
    "First-Occurence Word": list(combined_first_occurence.keys()),
    "Speaker": combined_speakers_list,
    "TimeStamp": combined_times_list
})

combined_df.to_csv(input("Output file name .csv: "), index = False)