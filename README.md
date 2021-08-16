# Counting-First-Occurrences
This script is used to count first occurrences of words per speaker within a meeting/conversaton, to answer questions such as who contributed the most amount of new words to the conversation?

Input to the script: 
- text files generated from https://otter.ai/home

Outputs from the script:
- a line graph showing the first occurrence count of each speaker overtime, based on the discrete timestamps given in the Otter transcripts
- a csv raw data file detailing every unique word spoken in the transcripts, when it was first spoken, and who said it (every row contains the first occurrence information of a unique word spoken in the conversation)

This script can take as many Otter.ai transcript text files as you want for input. It will merge all the timestamps across the transcripts into one timeline as if all the transcripts belong to the same conversation. So you can input only one text file to analyze one conversation, or multiple files to analyze overall first occurrence over all conversations for a project. Make sure to input the file names in chronological order of the sessions when prompted to input file names!

Sample input files are provided for you to experiment with the script.

!! Important: the script assumes your text files come with Otter.ai branding at the end. If they don't, please comment out "parsed_data = parsed_data[:-2]" at line 187
