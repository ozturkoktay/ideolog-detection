from datetime import datetime
import csv
import requests
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertForSequenceClassification
from flask import Flask, request, render_template
import codecs
from nltk.corpus import stopwords
import re
import torch
import numpy as np
import json
from universal_cleaner import cleanup, getLower
from utility import readJson
from validate_email import validate_email

stopWords = (stopwords.words('turkish'))
tokenizer = BertTokenizer.from_pretrained(
    'dbmdz/bert-base-turkish-128k-uncased', do_lower_case=True)
max_len = 512
model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-128k-uncased", num_labels=5,
                                                      output_attentions=False, output_hidden_states=False)
model.load_state_dict(torch.load(
    "../../models/23_09_2020_15_36_00___model_ideology.pt", map_location=torch.device('cpu')), strict=False)
model.eval()

SWEAR = []

with open("./kufurler/all_kufur_list.sorted.txt", encoding="utf-8") as handle:
    SWEAR = handle.readlines()
for i in range(0, len(SWEAR)):
    SWEAR[i] = SWEAR[i].replace("\n", " ")
    SWEAR[i] = SWEAR[i].strip()

phone_nums = readJson("config/phone_nums.json")
maps = readJson("config/configs.json")

CHAR_MAP = {
    "$": ("ş", "s"),
    "@": ("a", "o", "ö"),
    "4": ("a"),
    "7": ("t"),
    "3": ("e"),
    "5": ("s", "ş"),
    "1": ("l", "i", "ı"),
    "0": ("o", "ö"),
    "o": ("ö"),
    "c": ("ç"),
    "s": ("ş"),
    "u": ("ü"),
    "i": ("ı"),
    "g": ("ğ"),
}

LENGTH_MAP = {
    7: 4,
    6: 3,
    5: 2,
    4: 2,
    3: 1,
    2: 0,
    1: 0
}


def getAliases(text):
    output = [text]

    for c in CHAR_MAP:
        for v in CHAR_MAP[c]:
            t = text.replace(c, v)
            if t not in output:
                output.append(t)
    output = [o for o in output if len(o) > 0]
    return output


def findSwear(text):
    #: Replace some characters
    text = getLower(text)
    text = re.sub(r"[\-\_ ]+", " ", text).strip()
    #: Triple replacement
    text = re.sub(r"([a-zışüğçöA-ZÜĞİŞÇÖ])\1{2,}", r"\1\1", text)
    text = text.replace(".", " ")
    #: For each word in the given text
    for w in re.split(r"[^a-zA-Z0-9\$\@ışüğçöİŞÜĞÇÖ]", text):
        if len(w) > 0:
            #: For each alias in aliases
            for wi in getAliases(w):
                #: If exists in bloom, return
                if wi in SWEAR:
                    return True, w + "|" + wi
                if wi + "$" in SWEAR:
                    return True, w + "|" + wi
            #: Find the length
            l = 5
            if len(w) < 8:
                l = LENGTH_MAP[len(w)]
            if l > 0:
                #: Loop for l characters
                for i in range(l):
                    #: For each alias in aliases
                    for wi in getAliases(w[0:len(w) - i]):
                        #: If exits in bloom, return
                        if wi in SWEAR:
                            return True, w + "|" + wi
    #: Return default
    return False, None


def word_tokenize(sentence):
    acronym_each_dot = r"(?:[a-zğçşöüı]\.){2,}"
    acronym_end_dot = r"\b[a-zğçşöüı]{2,3}\."
    suffixes = r"[a-zğçşöüı]{3,}' ?[a-zğçşöüı]{0,3}"
    numbers = r"\d+[.,:\d]+"
    any_word = r"[a-zğçşöüı]+"
    punctuations = r"[a-zğçşöüı]*[.,!?;:]"
    word_regex = "|".join([acronym_each_dot,
                           acronym_end_dot,
                           suffixes,
                           numbers,
                           any_word,
                           punctuations])
    return re.compile("%s" % word_regex, re.I).findall(sentence)


def space_handler(s):
    s = s.replace("\n", " ")
    s = " ".join(s.split())
    return s


chan_words = {"slm": "selam", "efnedim": "efendim", "elmizde": "elimizde",
              "görüşe bilirsiniz": "görüşebilirsiniz", "ltfn": "lütfen", "lütfenn": "lütfen", "ltfen": "lütfen",
              "lüften": "lütfen",
              "ıade": "iade", "deyilmi": "değil mi", "deyil": "değil", "çözemezsiniz": "çözemezseniz",
              "edebilirmisiniz": "edebilir misiniz", "olmali": "olmalı", "kabuk": "kabul", "whatsup": "whatsapp",
              "efendım": "efendim", b"\xcc\x87".decode(): ""}


def replacer(x):
    for key in chan_words.keys():
        x = x.replace(key, chan_words[key])
    return x


def dataPrep(text):
    texts = []
    labels = [0]
    texts.append(text)
    test_texts = np.array(texts)
    test_labels = np.array(labels)
    input_ids = []
    attention_masks = []

    for text in test_texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(test_labels)

    batch_size = 7

    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    return prediction_dataloader


def pred(prediction_dataloader):
    predictions, true_labels = [], []

    for batch in prediction_dataloader:
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)
    preds_probable = []
    prediction_set = []

    for i in range(len(true_labels)):
        preds_probable.append(predictions[i])
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        prediction_set.append(pred_labels_i)
    return prediction_set


def predictor(text):
    asd = dataPrep(getLower(text))
    preds = pred(asd)
    if preds[0][0] == 0:
        return "Anarşizm"
    if preds[0][0] == 1:
        return "Komunizm"
    if preds[0][0] == 2:
        return "Liberalizm"
    if preds[0][0] == 3:
        return "Milliyetçilik"
    if preds[0][0] == 4:
        return "Muhafazakârlık"


#: Create application
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True


def readTextFile(filename):
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').strip('\r')
            if len(line) > 0:
                yield line
    f.close()


def determinor(text):
    if len(str(text)) == 0:
        return 'Yetersiz Cevap Verme'
    #: If rule based
    if re.search(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", str(text)):
        return "İletişim Bilgisi Paylaşan"
    if re.search(r"\b0?[\s\-]?5\d\d[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b", str(text)):
        return "İletişim Bilgisi Paylaşan"
    if re.search(r"\b0?[\s\-]?\d\d\d[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b", str(text)):
        return "İletişim Bilgisi Paylaşan"
    if re.search(r"[Tt][Rr]\s?\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{2}",
                 str(text)):
        return "IBAN Paylaşma"
    if re.search(
            r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
            str(text)):
        return "Link Paylaşma"
    if re.search(r"\@[a-zA-Z0-9]+\.(com|net|org)", str(text)):
        return "Mail Paylaşma"
    if re.search(r"[a-zA-Z0-9]+\.(com|net|org)", str(text)):
        return "Link Paylaşma"

    #: For mistyped
    text2 = str(text)
    text2 = re.sub(r"[^A-Za-zğüışçöÜĞİŞÇÖ0-9]", "", text2)
    if re.search(r"[ A-Za-zğüışçöÜĞİŞÇÖ]0?\d{10}[A-Za-zğüışçöÜĞİŞÇÖ ]", " " + text2 + " "):
        return "İletişim Bilgisi Paylaşan"

    if re.search(r"[ A-Za-zğüışçöÜĞİŞÇÖ]444\d{4}[A-Za-zğüışçöÜĞİŞÇÖ ]", " " + text2 + " "):
        return "İletişim Bilgisi Paylaşan"

    if re.search(r"[ A-Za-zğüışçöÜĞİŞÇÖ](90)?\d{10}[A-Za-zğüışçöÜĞİŞÇÖ ]", " " + text2 + " "):
        return "İletişim Bilgisi Paylaşan"
    if "ıfır beş yüz " in str(text):
        return "İletişim Bilgisi Paylaşan"

    #: Loop for each item
    for m in maps:
        parts = maps[m]

        for p in parts:
            if p in " " + str(text) + " ":
                return m

    return False


def cleanupNonDigits(text):
    return re.sub(r"[^0-9]", "", text)


def countdigits(text):
    return sum(c.isdigit() for c in text)


def countalphas(text):
    result = re.findall('[a-zA-ZşıüğçöâîûŞİÜĞÇÖÂÎÛ]', text)
    result = len(result)
    return result


def seekPhoneNumber(text):
    text = re.sub(r" +", " ", text)
    text = re.sub("[kK][pP][0-9]{11}|[k][p][0-9]{10}", "", text)
    parts = re.split(
        r"[a-zA-ZışüğçöÜĞİŞÇÖ\s][a-zA-ZışüğçöÜĞİŞÇÖ\s][a-zA-ZışüğçöÜĞİŞÇÖ\s]", text)

    parts = [p for p in parts if len(p) > 9]
    parts = [p.strip() for p in parts]

    parts = [p for p in parts if countdigits(p) >= 10]
    parts = [p for p in parts if countdigits(p) < 14]
    parts = [p for p in parts if countalphas(p) < 5]
    parts = [cleanupNonDigits(p) for p in parts]

    for i in range(len(parts)):
        if len(parts[i]) == 13 and parts[i][0:3] in "900":
            return True
        if len(parts[i]) == 12 and parts[i][0:2] == "90":
            return True
        if len(parts[i]) == 11 and parts[i][0:4] in phone_nums['special_start_zero'] + phone_nums[
                'operator_start_zero'] + phone_nums['city_start_zero']:
            return True
        if len(parts[i]) == 10 and parts[i][0:3] in phone_nums['special_start'] + phone_nums[
                'operator_start'] + phone_nums['city_start']:
            return True
    return False


def seekCardNumber(text):
    text = re.sub(r" +", " ", text)
    parts = re.split(r"[a-zA-ZışüğçöÜĞİŞÇÖ][a-zA-ZışüğçöÜĞİŞÇÖ]", text)
    parts = [p for p in parts if len(p) >= 16]
    parts = [p for p in parts if len(p) <= 26]
    parts = [p for p in parts if countdigits(p) == 16]
    parts = [cleanupNonDigits(p) for p in parts]

    return len(parts) > 0


def seekIBAN(text):
    text = re.sub(r" +", " ", text)
    parts = re.split(r"([a-zA-ZışüğçöÜĞİŞÇÖ][a-zA-ZışüğçöÜĞİŞÇÖ])", text)
    # parts = [p for p in parts if countdigits(p) == 24]
    # parts = [cleanupNonDigits(p) for p in parts]

    for i in range(len(parts)):
        if i > 0 and countdigits(parts[i]) == 24 and parts[i - 1].strip() in ['tr', 'TR', 'Tr', 'tR']:
            return True
    return False


def symbol_handler(text: str):
    if "@" in text:
        if validate_email(text, check_mx=False, verify=False):
            return False
        else:
            return True
    if "$" in text:
        if re.search(r"[0-9][$]|[$][0-9]", text):
            return False
        else:
            return True
    if "£" in text:
        if re.search(r"[0-9][£]|[£][0-9]", text):
            return False
        else:
            return True
    if "€" in text:
        if re.search(r"[0-9][€]|[€][0-9]", text):
            return False
        else:
            return True
    if "æ" in text:
        return True

    return False


LAST = ""


@app.route('/', methods=['GET', 'POST'])
def my_form():
    return render_template('index.html')


@app.route('/sentiment', methods=['GET', 'POST'])
def operate():
    global LAST
    sentence = request.form['input_text']
    # sentence = request.args.get("sentence", "nasılsınız efendim")
    sentence = str(sentence)
    sentence = sentence.strip()
    sentence = re.sub(r" +", " ", sentence)
    # sentence = getLower(sentence)

    spam = False
    if LAST == sentence:
        spam = True
    LAST = sentence

    # if countalphas(sentence) < 10: return jsonify({'info': 'çok kısa cümle, lütfen daha uzun cümle deneyiniz'})
    configs = ["lower", "circumflex", "htmltags", "spaces", "clean", "removal"]
    sent = predictor(cleanup(sentence, configs))
    sw, _ = findSwear(sentence)

    """
    if not sw:
        r = requests.post('http://192.168.118.195:5010/rootSentence', json={'text': sentence})
        r = r.json()['result']
        sw, _ = findSwear(r)
    """

    phone = seekPhoneNumber(sentence)
    card_number = seekCardNumber(sentence)
    iban = seekIBAN(sentence)
    determinator = determinor(sentence)
    symbol = symbol_handler(sentence)
    time = datetime.now()

    with open('./logs/request_logs.csv', mode='a', encoding='utf-8') as logs:
        log_writer = csv.writer(logs, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(
            [str(time), str(sentence), str(sent), str(determinator),  str(phone or card_number or iban), str(symbol), str(sw)])

    return render_template('result.html', info=str(determinator), input=sentence, sentiment=sent,
                           hidden_number=phone or card_number or iban, symbol=symbol, swear=sw)
