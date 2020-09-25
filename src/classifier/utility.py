#!usr/bin/env python
# -*- coding: utf-8 -*-

#: Import python modules
import os
import codecs
import json
import pickle
import numpy as np
from functools import reduce

# method: mape
# Mean absolute percentage error
# @val1, float: The first value
# @val2, float: The second value
# @return, float: The output percentage
# @completed
def mape( val1: float, val2: float ) -> float:
	total = val1 + val2
	diffr = abs(val1 - val2)
	return (diffr / total) * 2.0

# method: rescaleRange
# Rescales the range of given values, between zero and one
# @values, list: The input values
# @return, dict: The output value mapping dictionary
# @completed
def rescaleRange( values: list ) -> dict:
	#: Declare output
	output = {}
	#: Get min max
	mn = np.min( values )
	mx = np.max( values )
	#: Loop to create values
	if mx == mn:
		#: This means, there is only "ONE" value
		for v in values: output[v] = 0.0
		#: Return it
		return output
	#: Loop
	for v in values: output[v] = (v - mn) / (mx - mn)
	#: Return output
	return output

# method: checkFileExists
# Checks file exists
# @filename, str: The input filename to be checked
# @return, str: The checked filename
# @completed
def checkFileExists( filename: str ) -> str:
	#: Check file exists
	if os.path.isfile( filename ) == False:
		#: If not, raise error
		raise ValueError('Invalid filename', filename)
	#: Return if correct
	return filename

# method: loadData
# Loads the data from a source
# @source, str: The type of memory object [list, dictionary, memcachelist, memcachedict, bloomfilter, json]
# @path, str: The filename
# @return, object: The output object
# @completed
def loadData( source: str, path: str, config = None ):
	#: Declare output
	output = None
	#: Switch
	if source == 'list':
		#: Declare variables
		output = []
		#: Load from file
		for line in readTextFile( path ): output.append( line )
	elif source == 'key-json':
		#: Declare variables
		output = {}
		#: Load from file
		for line in readTextFile( path ): 
			#: Split the line
			line = line.split("\t")
			#: Parse json
			output[ line[0] ] = json.loads( line[1] )
	elif source == 'commalist':
		#: Declare variables
		output = {}
		#: Loop for each line
		for line in readTextFile( path ): 
			#: Get the parts
			line = line.split(",")
			#: Loop for each item
			for w in line:
				#: Assign all for it
				output[ w ] = line
	elif source == 'dict':
		#: Declare variables
		output = {}
		#: Load from file
		for line in readHashFile( path ): output[ line[0] ] = line[1]
	elif source == 'json':
		#: Load from file
		with open( path ) as f:
			#: Load json
			output = json.load(f)
	elif source == 'bloomfilter':
		#: Create the bloom filter
		output = BloomFilter(capacity= config['capacity'] , error_rate = config['error_rate'] )
		#: Load from file
		for line in readTextFile( path ): output.add( line )
	#: Return output
	return output

# method: readHashFile
# Reads the hash file
# @filename, str: The input filename
# @return, [tuple]: The output dictionary
# @completed
def readHashFile( filename: str ):
	filename = checkFileExists( filename )
	with codecs.open(filename, encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if len(line) > 0:
				line = line.split("\t")
				if len(line) == 2:
					yield line
	#: Close
	f.close()

# method: readTextFile
# Reads text file
# @filename, str: The input filename
# @return, [str]: The output
# @completed
def readTextFile( filename: str ):
	filename = checkFileExists( filename )
	with codecs.open(filename, encoding='utf-8') as f:
		for line in f:
			line = line.strip('\n').strip('\r')
			if len(line) > 0:
				yield line
	#: Close
	f.close()

# method: writePickle
# Writes a picke to a file
# @data, object: The object to be written
# @fileName, str: The filename to write
# @completed
def writePickle( data: object, fileName: str ):
	#: Open file for writing
	with open(fileName, 'wb') as f:
		#: Dump the object to the file as pickel
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
	#: Close
	f.close()

# method: readPickle
# Reads the pickle file
# @fileName, str: The input filename where to read object from
# @return, object: The returned object
# @completed
def readPickle( fileName: str ) -> object:
	#: Declare data
	data = None
	#: Read pickle
	with open(fileName, 'rb') as f:
		# The protocol version used is detected automatically, so we do not
		# have to specify it.
		data = pickle.load(f)
	#: Close file
	f.close()
	#: Return data
	return data

# method: writeJson
# Writes a json to a file
# @data, object: The object to be written
# @fileName, str: The filename to write
# @completed
def writeJson( data: object, fileName: str ):
	#: Open the file
	with open(fileName, 'w') as outfile:  
		json.dump(data, outfile)
	#: Close the file
	outfile.close()

# method: readJson
# Reads the json file
# @fileName, str: The input filename where to read object from
# @return, object: The returned object
# @completed
def readJson( fileName: str ) -> object:
	#: Declare data
	data = None
	#: Read pickle
	with open(fileName) as f:
		# The protocol version used is detected automatically, so we do not
		# have to specify it.
		data = json.load(f)
	#: Close file
	f.close()
	#: Return data
	return data

# method: appendTextFile
# Appends text to file
# @fileName, str: The name of the file
# @data, list: The data to be written
# @completed
def appendTextFile( fileName: str, data: list ):
	#: Open file
	file = codecs.open(fileName, "a", "utf-8")
	#: Write data
	for d in data: file.write( d + "\n")
	#: Close file
	file.close()

# method: writeTextFile
# Appends text to file
# @fileName, str: The name of the file
# @data, list: The data to be written
# @completed
def writeTextFile( fileName: str, data: list ):
	#: Open file
	file = codecs.open(fileName, "w", "utf-8")
	#: Write data
	for d in data: file.write( d + "\n")
	#: Close file
	file.close()

# variable: REG_ALL_NOTCHARS
# Not character match
REG_ALL_NOTCHARS = r"[^a-zşıüğçöâîûA-ZŞİÜĞÇÖÂÎÛ0-9 ]"

# method: replaceCircumflex
# Replaces circumflexex
# @input, str: The input string
# @return, str: The output replaced string
# @completed
def replaceCircumflex( input: str ) -> str:
	#: Map
	d = {
		u"Â":u"A", u"Î":u"I", u"Û":u"U", u"â":u"a", u"î":u"ı", u"û":u"u"
	}
	#: Replace
	input = reduce(lambda x, y: x.replace(y, d[y]), d, input)
	#: Return
	return input

# method: getLower
# Returns the lower
# @input, str: The input string
# @return, str: Lowered version
# @completed
def getLower( input: str ) -> str:
	#: Map
	d = {
	"Ş":"ş", "I":"ı", "Ü":"ü", "Ç":"ç", "Ö":"ö", "Ğ":"ğ", 
	"İ":"i", "Â":"â", "Î":"î", "Û":"û"
	}
	#: Replace
	input = reduce(lambda x, y: x.replace(y, d[y]), d, input)
	input = input.lower()
	#: Return
	return input
