#!usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# file: universal_cleaner.py
# A file consits of functions for cleaning and purifying an input text
# =============================================================================
#: Imports
from functools import reduce
import re
from html.parser import HTMLParser # HTML Stripping related operations


LOWERS = "abcdefghijklmnopqrstuvwxyzşıüğçöâîû"
UPPERS = "ABCDEFGJIJKLMNOPQRSTUVWXYZŞİÜĞÇÖÂÎÛ"
DIGITS = "0123456789"
LETTERS = LOWERS + UPPERS
CHARACTERS = LETTERS + DIGITS
PUNCTUATION = "…/,;.:!`'^+%&/()=?_-£#$½¾@<>{[]}|*~\""
SPACE = " \t\n\r"
SPECIAL = "«⟨»⟩¦―"
ALPHABET = CHARACTERS + PUNCTUATION + SPACE + SPECIAL
ALPHABET2 = CHARACTERS + PUNCTUATION
TOKENS = CHARACTERS + SPACE


# method: replaceMalformed
# Replaces malformed characters
# @input, str: The input string
# @return, str: The output replaced
# @completed
def replaceMalformed(input:str) -> str:
	input = re.sub(r"Ä±", "ı", input)
	input = re.sub(r"Ã§", "ç", input)
	input = re.sub(r"Ã¼", "ü", input)
	input = re.sub(r"ÅŸ", "ş", input)
	input = re.sub(r"Ã¶", "ö", input)
	input = re.sub(r"Ä°", "İ", input)
	input = re.sub(r"äÿ", "ğ", input)
	#: Return
	return input


# method: replaceCircumflex
# Replaces the circumflex
# @input, str: The input string
# @return, str: The output string
# @completed
def replaceCircumflex(input: str) -> str:
	d = {u"Â":u"A", u"Î":u"I", u"Û":u"U", u"â":u"a", u"î":u"ı", u"û":u"u"}
	input = reduce(lambda x, y: x.replace(y, d[y]), d, input)
	return input

# method: getLower
# Returns the lower
# @input, str: The input string
# @return, str: Lowered version
# @completed
def getLower(input: str) -> str:
	#: Map
	d = {
	"Ş":"ş", "I":"ı", "Ü":"ü",
	"Ç":"ç", "Ö":"ö", "Ğ":"ğ", 
	"İ":"i", "Â":"â", "Î":"î", 
	"Û":"û"
	}
	#: Replace
	input = reduce(lambda x, y: x.replace(y, d[y]), d, input)
	input = input.lower()
	#: Return
	return input

# method: internationalTransliterate
# International transliterate
# @input, str: The input string
# @return, str: The output string
# @completed
def internationalTransliterate(input: str) -> str:
	#: Replace the transliterated characters
	input = re.sub(r"[ṠṢṤṦṨẞŚŜŞŠȘⱾꞄ]", "S", input)
	input = re.sub(r"[ṪṬṮṰŢŤŦƬƮȚȾꞆ]", "T", input)
	input = re.sub(r"[ḁẚạảấầẩẫậắằẳẵặàáâãäåāăąǎǟǡǻȁȃȧⱥɐ]", "a", input)
	input = re.sub(r"[ṄṆṈṊÑŃŅŇƝǋǸȠꞐ]", "N", input)
	input = re.sub(r"[ḰḲḴĶƘǨⱩꝀꝂꝄ]", "K", input)
	input = re.sub(r"[ḋḍḏḑḓďđƌȡꝺɖɗ]", "d", input)
	input = re.sub(r"[ẎỲỴỶỸỾÝŶŸƳȲɎ]", "Y", input)
	input = re.sub(r"[ḔḖḘḚḜẸẺẼẾỀỂỄỆÈÉÊËĒĔĖĘĚƎƐȄȆȨɆ]", "E", input)
	input = re.sub(r"[ĵǰȷɉɟʄʝ]", "j", input)
	input = re.sub(r"[ẐẒẔŹŻŽƵȤⱫⱿ]", "Z", input)
	input = re.sub(r"[ẏẙỳỵỷỹỿýÿŷƴȳɏʎ]", "y", input)
	input = re.sub(r"[ṳṵṷṹṻụủứừửữựùúûüũūŭůűųưǔǖǘǚǜȕȗʉ]", "u", input)
	input = re.sub(r"[ḱḳḵķƙǩⱪꝁꝃꝅʞ]", "k", input)
	input = re.sub(r"[ḡĝğġģǥǧǵꝿɠɡ]", "ğ", input)
	input = re.sub(r"[ṫṭṯṱẗţťŧƫƭțȶⱦꞇʇʈ]", "t", input)
	input = re.sub(r"[ḕḗḙḛḝẹẻẽếềểễệèéêëēĕėęěǝȅȇȩɇⱸɘɛɜɝɞʚ]", "e", input)
	input = re.sub(r"[ĴɈ]", "J", input)
	input = re.sub(r"[ẀẂẄẆẈŴⱲ]", "W", input)
	input = re.sub(r"[ṽṿỽⱱⱴʋʌ]", "v", input)
	input = re.sub(r"[ḂḄḆƁƂɃ]", "B", input)
	input = re.sub(r"[ṡṣṥṧṩẛẜẝßśŝşšſșȿꞅʂ]", "s", input)
	input = re.sub(r"[ḢḤḦḨḪĤĦȞⱧⱵꞍ]", "H", input)
	input = re.sub(r"[ḉçćĉċčƈȼꜿɕ]", "c", input)
	input = re.sub(r"[ḊḌḎḐḒĎĐƉƊƋꝹ]", "D", input)
	input = re.sub(r"[ɋꝗʠ]", "q", input)
	input = re.sub(r"[ḃḅḇƀƃɓ]", "b", input)
	input = re.sub(r"[ḬỈỊÌÍÎÏĨĪĬĮİƗǏȈȊ]", "I", input)
	input = re.sub(r"[ḠĜĞĠĢƓǤǦǴꝽꝾ]", "G", input)
	input = re.sub(r"[ẑẓẕźżžƶȥɀⱬʐʑ]", "z", input)
	input = re.sub(r"[ṲṴṶṸṺỤỦỨỪỬỮỰÙÚÛÜŨŪŬŮŰŲƯǓǕǗǙǛȔȖɄ]", "U", input)
	input = re.sub(r"[ẁẃẅẇẉẘŵⱳʍ]", "w", input)
	input = re.sub(r"[ḞƑꝻ]", "F", input)
	input = re.sub(r"[ṙṛṝṟŕŗřȑȓɍⱹꞃɹɺɻɼɽɾɿ]", "r", input)
	input = re.sub(r"[ẋẍ]", "x", input)
	input = re.sub(r"[ṼṾỼƲɅ]", "V", input)
	input = re.sub(r"[ɊꝖ]", "Q", input)
	input = re.sub(r"[ḾṀṂƜⱮ]", "M", input)
	input = re.sub(r"[ḣḥḧḩḫẖĥħȟⱨⱶɥɦʮʯ]", "h", input)
	input = re.sub(r"[ḈÇĆĈĊČƇȻꜾ]", "C", input)
	input = re.sub(r"[ḶḸḺḼĹĻĽĿŁǈȽⱠⱢꝆꝈꞀ]", "L", input)
	input = re.sub(r"[ḟƒꝼ]", "f", input)
	input = re.sub(r"[ḭḯỉịìíîïĩīĭįıǐȉȋɨ]", "i", input)
	input = re.sub(r"[ḀẠẢẤẦẨẪẬẮẰẲẴẶÀÁÂÃÄÅĀĂĄǍǞǠǺȀȂȦȺⱯ]", "A", input)
	input = re.sub(r"[ṌṎṐṒỌỎỐỒỔỖỘỚỜỞỠỢÒÓÔÕÖØŌŎŐƆƟƠǑǪǬǾȌȎȪȬȮȰꝊꝌ]", "O", input)
	input = re.sub(r"[ṅṇṉṋñńņňŉƞǹȵꞑɲɳ]", "n", input)
	input = re.sub(r"[ẊẌ]", "X", input)
	input = re.sub(r"[ṔṖƤⱣꝐꝒ]", "P", input)
	input = re.sub(r"[ḿṁṃɯɰɱ]", "m", input)
	input = re.sub(r"[ḷḹḻḽĺļľŀłƚȴⱡꝇꝉꞁꞎɫɬɭ]", "l", input)
	input = re.sub(r"[ṕṗƥꝑꝓ]", "p", input)
	input = re.sub(r"[ṘṚṜṞŔŖŘȐȒɌⱤꞂ]", "R", input)
	input = re.sub(r"[ṍṏṑṓọỏốồổỗộớờởỡợòóôõöøōŏőơǒǫǭǿȍȏȫȭȯȱⱺꝋꝍɔɵ]", "o", input)
	#: Return
	return input

# method: correctNonUtf8
# Corrects non utf8 characters
# @input, str: The input string
# @return, str: The output replaced string
# @completed
def correctNonUtf8(input: str) -> str:
	#: Map
	d = {
	u"ý":u"ı", u"ð":u"ğ", u"þ":u"ş",
	u"Ð":u"Ğ", u"Ý":u"İ", u"Þ":u"Ş"
	}
	#: Replace
	input = reduce(lambda x, y: x.replace(y, d[y]), d, input)
	#: Return
	return input

# method: __transliterate__
# Replaces Non-English characters
# @input, str: The input string
# @return, str: The output replaced string
# @completed
def transliterate(input: str) -> str:
	#: Map
	d = {
	u"Ş":u"S",
	u"I":u"I",
	u"Ü":u"U",
	u"Ç":u"C",
	u"Ö":u"O",
	u"Ü":u"G",
	u"ş":u"s",
	u"ı":u"i",
	u"ü":u"u",
	u"ç":u"c",
	u"ö":u"o",
	u"ğ":u"g"
	}
	#: Replace
	input = reduce(lambda x, y: x.replace(y, d[y]), d, input)
	#: Return
	return input

# method: tokenPattern
# Returns the token pattern
# @text, str: The input string
# @return, str: The pattern
# @completed
def tokenPattern(text: str) -> str:
	#: Find the pattern
	pattern = re.sub(r"[ıçşüöğa-zâîû]+", "a", text)
	pattern = re.sub(r"[A-ZÂÎÛİŞÜĞÇÖ]+", "A", pattern)
	pattern = re.sub(r"[0-9]+", "0", pattern)
	#: Return pattern
	return pattern

# method: phonetic
# Returns the phonetic representation
# @text, str: The input text
# @return, str: The output text
# @completed
def phonetic(text: str) -> str:
	#: Replace letters
	text = re.sub(r"[bp]", "b", text)
	text = re.sub(r"[cçj]", "c", text)
	text = re.sub(r"[dt]", "d", text)
	text = re.sub(r"[fvw]", "f", text)
	text = re.sub(r"[gğkq]", "k", text)
	text = re.sub(r"[h]", "", text)
	text = re.sub(r"[ıîi]", "i", text)
	text = re.sub(r"[l]", "l", text)
	text = re.sub(r"[mn]", "m", text)
	text = re.sub(r"[oö]", "o", text)
	text = re.sub(r"[r]", "r", text)
	text = re.sub(r"[sşz]", "s", text)
	text = re.sub(r"[uüû]", "u", text)
	text = re.sub(r"[y]", "y", text)
	text = re.sub(r"[â]", "a", text)
	#: Multi characters
	text = re.sub(r"([iıçşüöğa-zâîû])\1+", r"\1", text)
	#: Return the text
	return text

# method: replaceEllipsis
# Replaces the ellipsis
# @input, str: The input string
# @return, str: The output replaced
# @completed
def replaceEllipsis(input: str) -> str:
	input = re.sub(r"\.{2,}", "…", input) 
	input = re.sub(r"(… …)+", "…", input) 
	input = re.sub(r"…+", "…", input)
	#: Return
	return input

# method: stripTags
# Strips the tags
# @string, str: The input html
# @return, str: The output string
# @completed
def stripTags(string:str, allowed_tags: str='') -> str:
	if allowed_tags != '':
		# Get a list of all allowed tag names.
		allowed_tags_list = re.sub(r'[\\/<> ]+', '', allowed_tags).split(',')
		allowed_pattern = ''
		for s in allowed_tags_list:
			if s == '':
			 continue;
			# Add all possible patterns for this tag to the regex.
			if allowed_pattern != '':
				allowed_pattern += '|'
			allowed_pattern += '<' + s + ' [^><]*>$|<' + s + '>|'
		# Get all tags included in the string.
		all_tags = re.findall(r'<]+>', string, re.I)
		for tag in all_tags:
			# If not allowed, replace it.
			if not re.match(allowed_pattern, tag, re.I):
				string = string.replace(tag, '')
	else:
		# If no allowed tags, remove all.
		string = re.sub(r'<[^>]*?>', '', string)
	#: Return
	return string

# method: firstLetterUpper
# Return first letter uppered
# @input, str: The input string
# @return, str: The output
# @completed
def firstLetterUpper(input: str) -> str:
	#: Lower case
	input = getLower(input)
	#: Return
	return input.title()

# method: replaceSpecial
# Replaces the special characters
# @input, str: The input string
# @return, str: The output string
# @completed
def replaceSpecial(input: str) -> str:
	#: Map
	d = {u"«":u"⟨", u"»":u"⟩", u"¦":u"|", u"―":u"-"}
	#: Replace
	input = reduce(lambda x, y: x.replace(y, d[y]), d, input)
	#: Return
	return input

# method: replaceSpaces
# Replaces the spaces
# @input, str: The input string
# @return, str: The output replaced
# @completed
def replaceSpaces(input: str) -> str:
	input = re.sub(r"\n+$", "", input)
	input = re.sub(r"^\n+", "", input)
	input = re.sub(r"\n{3,}", "\n\n", input)
	#input = re.sub(r"\h+$", "", input)
	#input = re.sub(r"^\h+", "", input)
	input = re.sub(r"[\t ]+", " ", input)
	# input = re.sub(r"\r", "", input)
	return input


def trim(input: str) -> str:
	input = re.sub(r" +", " ", input)
	return input.strip()


def remove_repeated_word(sentence: str) -> str:
    """
    Remove repeated words. E.x: Çok çok güzel güzel bilgisayar -> çok güzel bilgisayar
	
    :param sentence: given text
    :type sentence: str
    :return: removed duplicated words
    :rtype: str
    """
    text = re.sub(r'\b(\w+\s*)\1{1,}', '\\1', sentence)
    return text 


def consecutive_removal(text: str) -> str:
    text = re.sub(r'(.)\1+', r'\1\1', text)
    return text


def clean(sentence: str) -> str:
    """
    Clean given text for Turkish language

    :param sentence: given string sentence
    :type sentence: str
    :return: cleaned text
    :rtype: str
    """
    sentence = consecutive_removal(getLower(sentence))
    sentence = re.sub(r"[^abcdefghijklmnopqrstuvwxyzşıüğçö…/:.+_-₺£$@0-9 ]", " ", sentence)
    sentence = sentence.strip()
    return sentence


def cleanup(text: str, config: list) -> str:
    """
    Main method for applying for all methods 

    :param text: given text for cleaning
    :type text: str
    :param config: method list
    :type config: list
    :return: cleaned text for config list
    :rtype: str
    """
    if "circumflex" in config: text = replaceCircumflex(text)
    if "ellipsis" in config: text = replaceEllipsis(text)
    if "lower" in config: text = getLower(text)
    if "internationaltransliterate" in config: text = internationalTransliterate(text)
    if "correctnonutf8" in config: text = correctNonUtf8(text)
    if "transliterate" in config: text = transliterate(text)
    if "tokenpattern" in config: text = tokenPattern(text)
    if "phonetic" in config: text = phonetic(text)
    if "malformed" in config: text = replaceMalformed(text)
    if "htmltags" in config: text = stripTags(text)
    if "title" in config: text = firstLetterUpper(text)
    if "special" in config: text = replaceSpecial(text)
    if "spaces" in config: text = replaceSpaces(text)
    if "removal" in config: text = consecutive_removal(text)
    if "clean" in config: text = clean(text)
    return trim(text)