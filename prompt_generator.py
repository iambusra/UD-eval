import re
import json
from pathlib import Path

# Global warnings list for features not printed.
global_warnings = []

# Global set for tracking handled features.
global_handled_features = set()

# Set of known feature keys.
known_feature_keys = {
    "VerbForm", "NumType", "Gender", "Reflex", "PronType",
    "Number", "Person", "Person[psor]", "Number[psor]", "Case",
    "Abbr", "Typo", "Foreign", "Poss", "Echo",
    "Polite", "Polite[abs]", "Polite[dat]", "Polite[erg]",
    "Dialect", "ExtPos", "Voice", "Polarity", "Mood", "Aspect", "Tense", "Evident",
    "Animacy", "Definite", "Degree", "Gender[abs]", "Person[abs]", "Number[abs]",
    "Gender[erg]", "Person[erg]", "Number[erg]",
    "Gender[dat]", "Person[dat]", "Number[dat]", "PartType", "Degree", "Style", "NumForm",
    "AdpType", "AdvType", "Echo", "Form", "NounType", "PrepForm"
}

# Mapping dictionaries for natural language equivalents
upos_map = {
    "ADJ": "an adjective",
    "ADP": "an adposition",
    "ADV": "an adverb",
    "AUX": "an auxiliary verb",  # may be updated via VerbForm
    "CCONJ": "a coordinating conjunction",
    "DET": "a determiner",
    "INTJ": "an interjection",
    "NOUN": "a noun",
    "NUM": "a numeral",
    "PART": "a particle",
    "PRON": "a pronoun",
    "PROPN": "a proper noun",
    "PUNCT": "a punctuation",
    "SCONJ": "a subordinating conjunction",
    "SYM": "a symbol",
    "VERB": "a verb",
    "X": "an undefined category"
}

pron_type_map = {
    "Art": "article",
    "Dem": "demonstrative",
    "Emp": "emphatic",
    "Ind": "indefinite",
    "Int": "interrogative",
    "Neg": "negative",
    "Prs": "personal",
    "Rcp": "reciprocal",
    "Rel": "relative",
    "Tot": "total",
    "Loc": "locative"
}

degree_map = {
    "Abs": "absolute",
    "Cmp": "comparative",
    "Pos": "positive",
    "Sup": "superlative",
    "Cmp,Sup": "comparative-superlative"
}

part_type_map = {
    "Ad": "adverbial",
    "Cmpl": "complementizer",
    "Comp": "comparative",
    "Cop": "copular",
    "Deg": "degree",
    "Inf": "infinitive",
    "Int": "interrogative",
    "Num": "numeral",
    "Pat": "patronym",
    "Sup": "superlative",
    "Vb": "verbal",
    "Voc": "vocative"
}


numtype_map = {
    "Card": "a cardinal numeral",
    "Ord": "an ordinal numeral",
    "Dist": "a distributive numeral",
    "Frac": "a fractional numeral",
    "Mult": "a multiplicative numeral"
}

# a numeral can have BOTH numform and numtype, handle accordingly
numform_map = {
    "Digit": "in digits",
    "Word": "in words",
    "Combi": "in digits with a suffix",
    "Roman": "in Roman numerals",
}

# Universal POS-modifier maps
adptype_map = {
    "Post": "postposition",
}

advtype_map = {
    "Deg": "degree",
}

echo_map = {
    "Rdp": "reduplicative"
}

verbform_map = {
    "Conv": "a converb",
    "Part": "a participle",
    "Vnoun": "a verbal noun",
    "Fin": "a finite verb",
    "Inf": "an infinitive verb",
    "Ger": "a gerund",
    "Cop": "a copula",
}

gender_map = {
    "Fem": "feminine",
    "Masc": "masculine",
    "Neut": "neuter",
    "Fem,Masc": "feminine or masculine"
}

case_map = {
    "Nom": "nominative",
    "Abl": "ablative",
    "Acc": "accusative",
    "Dat": "dative",
    "Equ": "equative",
    "Gen": "genitive",
    "Ins": "instrumental",
    "Loc": "locative",
    "Abs": "absolutive",
    "All": "allative",
    "Ben": "benefactive",
    "Cau": "causative",
    "Com": "comitative",
    "Erg": "ergative",
    "Ess": "essive",
    "Ine": "inessive",
    "Lat": "lative",
    "Par": "partitive",
    "Voc": "vocative",
    "Acc,Gen": "accusative-genitive syncretic",
    "Acc,Dat": "accusative-dative syncretic",
    "Acc,Erg": "accusative-ergative syncretic",
    "Acc,Ine": "accusative-inessive syncretic",
    "Acc,Ins": "accusative-instrumental syncretic",
    "NomAcc": "nominative-accusative syncretic"
}

number_map = {
    "Sing": "singular",
    "Plur": "plural"
}

person_map = {
    "0": "autonomous",
    "1": "first",
    "2": "second",
    "3": "third"
}

polarity_map = {
    "Neg": "negative",
    "Pos": "positive"
}

aspect_map = {
    "Hab": "habitual aspect"
}

aspect_map_new = {
    "Hab": "habitual aspect",
    "Imp": "imperfect aspect",
    "Perf": "perfect aspect",
    "Prog": "progressive aspect",
    "Prosp": "prospective aspect"
}

tense_map = {
    "Fut": "future tense",
    "Past": "past tense",
    "Pqp": "pluperfect tense",
    "Pres": "present tense"
}

evident_map = {
    "Fh": "direct evidential marker",
    "Nfh": "indirect evidential marker"
}

mood_map = {
    "Cnd": "conditional mood",
    "Des": "desiderative mood",
    "DesPot": "desiderative and potential mood markers",
    "Gen": "generalized modality",
    "GenNec": "general necessitative mood",
    "GenPot": "general potential mood",
    "Imp": "imperative mood",
    "Ind": "indicative mood",
    "Nec": "necessitative mood",
    "Opt": "optative mood",
    "Pot": "potential mood",
    "Sub": "subjunctive mood",
    "Int": "interrogative mood",
    "Cnd,Int": "conditional interrogative mood",
    "Imp,Int": "imperative interrogative mood"
}

definite_map = {
    "Def": "definite",
    "Ind": "indefinite",
    "Indef": "indefinite"
}

animacy_map = {
    "Anim": "animate",
    "Inan": "inanimate"
}

# Style feature → first‐sentence modifier
style_map = {
    "Arch": "in archaic language",
    "Coll": "in colloquial language",
    "Expr": "in expressive language",
    "Slng": "in slang language",
    "Vrnc": "in vernacular language"
}

# Extra features Group 1
group1_features = {
    "Abbr": ("Yes", "an abbreviation"),
    "Typo": ("Yes", "a mistyped word"),
    "Foreign": ("Yes", "a foreign word"),
    "Poss": ("Yes", "a possessive"),
    "Echo": ("Rdp", "a reduplication")
}

polite_map = {
    "Infm": "in informal register",
    "Form": "in formal register"
}

polite_variant_map = {
    "Polite[abs]": {"Infm": "in informal register, agreeing with absolutive argument"},
    "Polite[dat]": {"Infm": "in informal register, agreeing with dative argument"},
    "Polite[erg]": {"Infm": "in informal register, agreeing with ergative argument"}
}
dialect_map = {
    "Connaught": "in Connaught dialect",
    "Munster": "in Munster dialect",
    "Ulster": "in Ulster dialect"
}
extpos_map = {
    "ADP": "an adposition-like expression",
    "ADV": "an adverb-like expression",
    "CCONJ": "a coordinating conjunction-like expression",
    "PRON": "a pronoun-like expression",
    "SCONJ": "a subordinator-like expression"
}

form_map = {
    "Direct": "direct",
    "Ecl": "eclipsis",
    "Emp": "emphatic",
    "HPref": "h-prefix",
    "Indirect": "indirect",
    "Len": "lenition",
    "VF": "vowel form",
    "Ecl,Emp": "emphatic eclipsis"
}

noun_type_map = {
    "NotSlender": "with broad consonants",
    "Slender": "with slender consonants",
    "Strong": "with strong plurals",
    "Weak": "with weak plurals"
}

prepform_map = {
    "Cmpd": "compound"
}


dependency_map = {
    "acl": "clausal modifier of noun",
    "acl:recl": "relative clause modifier",
    "advcl": "adverbial clause modifier",
    "advmod": "adverbial modifier",
    "advmod:emph": "intensifier",
    "amod": "adjectival modifier",
    "appos": "appositional modifier",
    "aux": "auxiliary",
    "aux:pass": "passive auxiliary",
    "aux:q": "question particle",
    "case": "case marking",
    "cc": "coordinating conjunction",
    "cc:preconj": "preconjunct",
    "ccomp": "clausal complement",
    "clf": "classifier",
    "compound": "compound",
    "compound:ext": "extent and descriptive verb compound",
    "compound:lvc": "light verb construction",
    "compound:redup": "reduplicated compounds",
    "conj": "conjunct",
    "cop": "copula",
    "csubj": "clausal subject",
    "csubj:outer": "clausal subject of outer clause",
    "dep": "unspecified",
    "dep:der": "derivational suffix",
    "det": "determiner",
    "discourse": "discourse element",
    "discourse:sp": "sentence particle",
    "discourse:q": "discourse particle for questions",
    "dislocated": "dislocated element",
    "fixed": "multi-word expression",
    "flat": "name",
    "flat:name": "name",
    "flat:foreign": "foreign word",
    "goeswith": "goes with",
    "iobj": "indirect object",
    "list": "list",
    "mark": "marker",
    "mark:adv": "manner adverbializer",
    "mark:rel": "relativizer",
    "nmod": "nominal modifier",
    "nmod:part": "nominal modifier indicating part-whole relations",
    "nmod:poss": "possessive nominal modifier",
    "nsubj": "nominal subject",
    "nsubj:outer": "nominal subject of outer clause",
    "nummod": "numeric modifier",
    "obj": "direct object",
    "obl": "oblique",
    "obl:tmod": "temporal modifier",
    "obl:patient": "object in BA construction",
    "orphan": "remnant in ellipsis",
    "parataxis": "parataxis",
    "punct": "punctuation",
    "vocative": "vocative",
    "xcomp": "open clausal complement"
}

# --- Helper Functions ---
def parse_feats(feat_str):
    feats = {}
    if feat_str == "_" or feat_str.strip() == "":
        return feats
    for feat in feat_str.split("|"):
        if "=" in feat:
            key, val = feat.split("=", 1)
            feats[key] = val
    return feats

def validate_feature_maps(conllu_content):
    """
    Validate that all features and their values in a CoNLL-U content have mappings.

    Args:
        conllu_content (str): CoNLL-U format content

    Returns:
        list: List of error messages for missing mappings
    """
    errors = []

    for line in conllu_content.split("\n"):
        if line.startswith("#") or not line.strip():
            continue

        # Parse the CoNLL-U line
        fields = line.strip().split("\t")
        if len(fields) < 6:
            continue

        # Get the FEATS field
        feats_str = fields[5]
        if feats_str == "_":
            continue

        # Get the UPOS field
        upos = fields[3]

        # Parse features
        feats = parse_feats(feats_str)

        # Check each feature and its value
        for feat, value in feats.items():
            # First check if feature is known
            if feat not in known_feature_keys:
                errors.append(f"Unknown feature '{feat}'")
                continue

            # Now check value mappings based on feature type
            if feat == "VerbForm" and value not in verbform_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Gender" and value not in gender_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Case" and value not in case_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Number" and value not in number_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Person" and value not in person_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Polarity" and value not in polarity_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Aspect" and value not in aspect_map_new:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Tense" and value not in tense_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Evident" and value not in evident_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Mood" and value not in mood_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Definite" and value not in definite_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Style" and value not in style_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "PronType" and value not in pron_type_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Degree" and value not in degree_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "NumType" and value not in numtype_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "NumForm" and value not in numform_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "PartType" and value not in part_type_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "AdpType" and value not in adptype_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "AdvType" and value not in advtype_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Echo" and value not in echo_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Polite" and value not in polite_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "Form" and value not in form_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "NounType" and value not in noun_type_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")
            elif feat == "PrepForm" and value not in prepform_map:
                errors.append(f"Missing value mapping: '{value}' for feature '{feat}'")

    return errors

# Modified join_phrases: join items with ", " then final join with " and " (no comma before "and").
def join_phrases(phrases):
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]

# For NOUN, PROPN, ADJ, NUM, ADP, and now PRON we print detailed morphology.
def describe_case_agreement_features(feats):
    phrases = []
    person_part = None

    if "Number" in feats and "Number" not in global_handled_features:
        global_handled_features.add("Number")
        number_phrase = number_map.get(feats["Number"], feats["Number"])
        if number_phrase not in phrases:
            phrases.append(number_phrase)


    if "Person" in feats and "Person" not in global_handled_features:
        global_handled_features.add("Person")
        person_text = person_map.get(feats["Person"], feats["Person"])
        person_part = f"marked for {person_text} person"

    if person_part:
        phrases.append(person_part)

    if "Person[psor]" in feats and "Number[psor]" in feats:
        global_handled_features.add("Person[psor]")
        global_handled_features.add("Number[psor]")
        phrases.append(
            f"carries a {person_map.get(feats['Person[psor]'])} person {number_map.get(feats['Number[psor]'])} possessive marker"
        )

    if "Case" in feats and "Case" not in global_handled_features:
        global_handled_features.add("Case")
        phrases.append(f"in {case_map.get(feats['Case'], feats['Case'])} case")

    if "Animacy" in feats and "Animacy" not in global_handled_features:
        global_handled_features.add("Animacy")
        phrases.append(animacy_map[feats["Animacy"]])

    return f"It is {join_phrases(phrases)}." if phrases else ""

# Extra features for tokens not covered above.
def describe_extra_features(feats, upos=None):
    ordered = []

    # Group-1 extras
    for key in ["Abbr", "Typo", "Foreign", "Poss", "Echo"]:
        if key in feats and feats[key] == group1_features[key][0]:
            global_handled_features.add(key)
            ordered.append(group1_features[key][1])

    # Politeness
    if "Polite" in feats:
        val = feats["Polite"]
        if val in polite_map:
            global_handled_features.add("Polite")
            ordered.append(polite_map[val])
    for key in polite_variant_map:
        if key in feats:
            val = feats[key]
            if val in polite_variant_map[key]:
                global_handled_features.add(key)
                ordered.append(polite_variant_map[key][val])

    # Dialect
    if "Dialect" in feats:
        val = feats["Dialect"]
        if val in dialect_map:
            global_handled_features.add("Dialect")
            ordered.append(dialect_map[val])

    # External-POS
    if "ExtPos" in feats:
        val = feats["ExtPos"]
        if val in extpos_map:
            global_handled_features.add("ExtPos")
            ordered.append(extpos_map[val])


    # Polarity on content words
    if "Polarity" in feats and "Polarity" not in global_handled_features:
        global_handled_features.add("Polarity")
        ordered.append("It has " + polarity_map.get(feats["Polarity"], feats["Polarity"]) + " polarity")

    # For everything *but* core morpho-agreement words, also allow number/person etc.
    if upos not in ["NOUN", "PROPN", "ADJ", "NUM", "ADP", "PRON", "VERB", "AUX"]:
        if "Number" in feats:
            global_handled_features.add("Number")
            ordered.append("in " + number_map.get(feats["Number"], feats["Number"]))
        if "Person" in feats:
            global_handled_features.add("Person")
            ordered.append("marked for " + person_map.get(feats["Person"], feats["Person"]) + " person")
        if "Person[psor]" in feats and "Number[psor]" in feats:
            global_handled_features.add("Person[psor]")
            global_handled_features.add("Number[psor]")
            ordered.append(
                f"carries a {person_map.get(feats['Person[psor]'], feats['Person[psor]'])} person "
                f"{number_map.get(feats['Number[psor]'], feats['Number[psor]'])} possessive marker"
            )
        if "Case" in feats:
            global_handled_features.add("Case")
            ordered.append("in " + case_map.get(feats["Case"], feats["Case"]) + " case")

    # ————  **NEW GUARD** ————
    # only append Animacy if it wasn’t already handled above
    if "Animacy" in feats and "Animacy" not in global_handled_features:
        anim = feats["Animacy"]
        if anim in animacy_map:
            global_handled_features.add("Animacy")
            ordered.append(animacy_map[anim])

    # Voice on *any* POS
    if "Voice" in feats and "Voice" not in global_handled_features:
        global_handled_features.add("Voice")
        voice_val = feats["Voice"]
        # same mapping as in describe_verb_features
        voice_map = {
            "Cau": "causative voice",
            "CauCau": "double causative voice",
            "CauPass": "causative and passive voice markers",
            "Pass": "passive voice",
            "PassPass": "double passive voice",
            "PassRfl": "reflexive and passive voice markers",
            "Rcp": "reciprocal voice",
            "Rfl": "reflexive voice",
            "Act": "active voice"
        }
        if voice_val in voice_map:
            ordered.append(voice_map[voice_val])

    # Aspect on *any* POS
    if "Aspect" in feats and "Aspect" not in global_handled_features:
        global_handled_features.add("Aspect")
        a = feats["Aspect"]
        if a in aspect_map_new:
            ordered.append(aspect_map_new[a])

    # Format into a sentence/fragment
    if ordered:
        combined = " and ".join(ordered)
        lower_comb = combined.lower().strip()
        if lower_comb.startswith("carries"):
            # e.g. "carries a X and a Y"
            return "It " + combined + "."
        # catch both "has ..." and "It has ..." (plus animate/inanimate)
        elif lower_comb.startswith(("has", "it has", "animate", "inanimate")):
            # ensure exactly one period
            return combined if combined.endswith(".") else combined + "."
        else:
            return "It is " + combined + "."
    else:
        return ""


def get_ordinal(n):
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def get_definite_phrase(original_description, feats):
    """
    Inject definiteness modifier correctly into description.
    """
    if "Definite" not in feats:
        return original_description

    global_handled_features.add("Definite")
    definite_value = feats["Definite"]
    modifier = definite_map.get(definite_value, definite_value)

    # Remove existing article and add new
    base = original_description
    if base.startswith("a "):
        base = base[2:]
    elif base.startswith("an "):
        base = base[3:]

    new_phrase = f"{modifier} {base}"
    article = "an" if new_phrase[0].lower() in "aeiou" else "a"
    return f"{article} {new_phrase}"

def get_pos_description(upos, feats):
    vowels = set("aeiou")

    parts = []

    # Reflexive (as first prefix)
    if "Reflex" in feats and feats["Reflex"] == "Yes" and "Reflex" not in global_handled_features:
        parts.append("reflexive")
        global_handled_features.add("Reflex")

    # Gender (comes second)
    if "Gender" in feats and "Gender" not in global_handled_features:
        parts.append(gender_map[feats["Gender"]])
        global_handled_features.add("Gender")

    # Form (immediately after Gender)
    if "Form" in feats and "Form" not in global_handled_features:
        form_val = form_map.get(feats["Form"], feats["Form"].lower())
        parts.append(form_val)
        global_handled_features.add("Form")

    # Degree
    if "Degree" in feats and "Degree" not in global_handled_features:
        parts.append(degree_map[feats["Degree"]])
        global_handled_features.add("Degree")

    # NumType
    if "NumType" in feats and "NumType" not in global_handled_features:
        nt = numtype_map[feats["NumType"]].replace("a ", "").replace("an ", "").replace(" numeral", "")
        parts.append(nt)
        global_handled_features.add("NumType")

    # AdpType
    if "AdpType" in feats and upos in ("ADP", "NOUN", "ADV", "ADJ") and "AdpType" not in global_handled_features:
        core = adptype_map.get(feats["AdpType"], feats["AdpType"].lower())
        parts.append(core)
        global_handled_features.add("AdpType")

    # AdvType
    if "AdvType" in feats and upos in ("ADP", "NOUN", "ADV", "ADJ") and "AdvType" not in global_handled_features:
        core = advtype_map.get(feats["AdvType"], feats["AdvType"].lower())
        parts.append(core)
        global_handled_features.add("AdvType")

    # PronType
    if "PronType" in feats and "PronType" not in global_handled_features:
        core = pron_type_map.get(feats["PronType"], feats["PronType"].lower())
        parts.append(core)
        global_handled_features.add("PronType")

    # VerbForm (ALWAYS just before the noun)
    if "VerbForm" in feats and "VerbForm" not in global_handled_features:
        core = verbform_map.get(feats["VerbForm"], feats["VerbForm"].lower()).replace("a ", "").replace("an ", "")
        parts.append(core)
        global_handled_features.add("VerbForm")

    # PrepForm (as a prefix before PartType/noun label)
    if "PrepForm" in feats and "PrepForm" not in global_handled_features:
        prepform = prepform_map.get(feats["PrepForm"], feats["PrepForm"].lower())
        parts.append(prepform)
        global_handled_features.add("PrepForm")


    # PartType (as a prefix before noun label)
    if "PartType" in feats and "PartType" not in global_handled_features:
        parttype = part_type_map.get(feats["PartType"], feats["PartType"].lower())
        parts.append(parttype)
        global_handled_features.add("PartType")

    # Noun label (always last in core phrase)
    noun = upos_map.get(upos, upos.lower())
    if " " in noun:
        noun = noun.split(" ", 1)[1]  # drop "a"/"an"
    parts.append(noun)

    # Compose the main phrase
    phrase = " ".join(parts)
    article = "an" if phrase[0].lower() in vowels else "a"

    # NounType as suffix (if present)
    if "NounType" in feats and "NounType" not in global_handled_features:
        noun_type = noun_type_map.get(feats["NounType"], feats["NounType"])
        phrase = f"{phrase} {noun_type}"
        global_handled_features.add("NounType")


    # NumForm (for NUM, can come at the end)
    numform_phrase = ""
    if "NumForm" in feats and "NumForm" not in global_handled_features:
        numform_phrase = numform_map.get(feats["NumForm"], feats["NumForm"])
        global_handled_features.add("NumForm")
    # ----------------------------------------

    return get_definite_phrase(f"{article} {phrase}", feats)


def describe_verb_features(feats):
    parts = []
    if "Voice" in feats:
        global_handled_features.add("Voice")
        voice = feats["Voice"]
        voice_map = {
            "Cau": "causative voice",
            "CauCau": "double causative voice",
            "CauPass": "causative and passive voice markers",
            "Pass": "passive voice",
            "PassPass": "double passive voice",
            "PassRfl": "reflexive and passive voice markers",
            "Rcp": "reciprocal voice",
            "Rfl": "reflexive voice",
            "Act": "active voice"
        }
        if voice in voice_map:
            parts.append(voice_map[voice])
    if "Polarity" in feats:
        global_handled_features.add("Polarity")
        polarity = feats["Polarity"]
        if polarity in polarity_map:
            parts.append(polarity_map[polarity] + " polarity")
    if "Mood" in feats:
        global_handled_features.add("Mood")
        mood = feats["Mood"]
        if mood in mood_map:
            parts.append(mood_map[mood])
    if "Aspect" in feats:
        global_handled_features.add("Aspect")
        aspect = feats["Aspect"]
        if aspect in aspect_map_new:
            parts.append(aspect_map_new[aspect])
    if "Tense" in feats:
        global_handled_features.add("Tense")
        tense = feats["Tense"]
        if tense in tense_map:
            parts.append(tense_map[tense])
    if "Evident" in feats:
        global_handled_features.add("Evident")
        evident = feats["Evident"]
        if evident in evident_map:
            parts.append(evident_map[evident])
    if parts:
        return "It has " + join_phrases(parts) + "."
    else:
        return ""

def describe_argument_agreement_features(feats):
    """
    Look for Gender[role], Person[role], Number[role] for role in {abs, erg, dat}
    and build sentences like:
      “It has a feminine third person singular ergative argument.”
      or
      “It has a feminine third person singular ergative argument and a masculine third person plural dative argument.”
    """
    roles = []
    for role in ("abs", "erg", "dat"):
        gk = f"Gender[{role}]"
        pk = f"Person[{role}]"
        nk = f"Number[{role}]"
        if gk in feats or pk in feats or nk in feats:
            parts = []
            if gk in feats:
                parts.append(gender_map[feats[gk]])
                global_handled_features.add(gk)
            if pk in feats:
                parts.append(f"{person_map[feats[pk]]} person")
                global_handled_features.add(pk)
            if nk in feats:
                parts.append(number_map[feats[nk]])
                global_handled_features.add(nk)
            role_name = case_map[role.capitalize()]  # “abs”→“Abs”→“absolutive”
            sent = " ".join(parts + [role_name, "argument"])
            roles.append(f"a {sent}")
    if not roles:
        return ""
    return f"It has {' and '.join(roles)}."

def describe_token(token_fields, sent_id=None):
    global global_handled_features
    global_handled_features = set()  # Reset for each token

    index = token_fields[0]
    ordinal = get_ordinal(index)
    feat_str = token_fields[5]
    feats = parse_feats(feat_str)

    for feat_key in feats.keys():
        if feat_key not in known_feature_keys:
            global_warnings.append(f"Unknown feature key '{feat_key}' found in token {ordinal} in sentence {sent_id}")

    lemma = token_fields[2]
    upos = token_fields[3]
    head = token_fields[6]
    deprel = token_fields[7]

    pos_description = get_pos_description(upos, feats)

    # ——— inject Style into the first sentence ———
    if "Style" in feats:
        style_desc = style_map.get(feats["Style"])
        if style_desc:
            global_handled_features.add("Style")
            # e.g. "a noun" → "a noun in archaic language"
            pos_description = f"{pos_description} {style_desc}"
    # ————————————————————————————————————————

    # If pos_description starts with "its syntactic category" or another clause, don't add "is"
    if pos_description.startswith(("its syntactic category", "this token", "the token")):
        core_first_sentence = f'The {ordinal} token has "{lemma}" as its lemma and {pos_description}.'
    else:
        core_first_sentence = f'The {ordinal} token has "{lemma}" as its lemma and is {pos_description}.'


    # Print verbal features for all POS except PUNCT
    if upos != "PUNCT":
        verb_feat_sentence = describe_verb_features(feats)
    else:
        verb_feat_sentence = ""

    # Agreement sentence ONLY for verbs and aux (same as before)
    agreement_sentence = ""
    has_agreement = False
    if upos in ["VERB", "AUX"]:
        if "Person" in feats and "Number" in feats:
            global_handled_features.add("Person")
            global_handled_features.add("Number")
            has_agreement = True
            agreement_sentence = f"It has {person_map.get(feats['Person'])} person {number_map.get(feats['Number'])} agreement."
        elif "Person" in feats:
            global_handled_features.add("Person")
            has_agreement = True
            agreement_sentence = f"It has {person_map.get(feats['Person'])} person agreement."
        elif "Number" in feats:
            global_handled_features.add("Number")
            has_agreement = True
            agreement_sentence = f"It has {number_map.get(feats['Number'])} agreement."

    # For verbs/aux: avoid duplicate printing of Number/Person in case-agreement
    if upos in ["VERB", "AUX"]:
        morph_feats = {k: v for k, v in feats.items() if k not in {"Person"} if not (has_agreement and k == "Number")}
        morph_sentence = describe_case_agreement_features(morph_feats)
    else:
        morph_sentence = describe_case_agreement_features(feats)

    extra_sentence = describe_extra_features(feats, upos)

    output_text = core_first_sentence
    if verb_feat_sentence:
        output_text += " " + verb_feat_sentence
    if agreement_sentence:
        output_text += " " + agreement_sentence
    if morph_sentence:
        output_text += " " + morph_sentence
    if extra_sentence:
        output_text += " " + extra_sentence


    # --- new agreement mapping ---
    arg_agreement = describe_argument_agreement_features(feats)
    if arg_agreement:
        output_text += " " + arg_agreement

    if head == "0":
        dep_sentence = "It is the root node."
    else:
        dep_sentence = f"Its head is the {get_ordinal(head)} token and its dependency label is {dependency_map.get(deprel, deprel)}."

    output_text += " " + dep_sentence

    for attr, val in feats.items():
        # skip anything we've already printed
        if attr in global_handled_features:
            continue

        if attr in group1_features and feats[attr] == group1_features[attr][0]:
            if group1_features[attr][1].lower() in output_text.lower():
                continue
        if attr in {"Person[psor]", "Number[psor]"}:
            if "Person[psor]" in feats and "Number[psor]" in feats:
                phrase = (
                    f"carries a {person_map[feats['Person[psor]']]} person "
                    f"{number_map[feats['Number[psor]']]} possessive marker"
                )
                if phrase.lower() in output_text.lower():
                    continue
        if upos == "NUM" and attr == "NumType":
            continue
        if attr == "PronType":
            if pron_type_map.get(val, val).lower() in output_text.lower():
                continue
        if attr == "VerbForm":
            if verbform_map.get(val, val).lower() in output_text.lower():
                continue

        # if we get here, nothing printed it
        global_warnings.append(
            f"Feature {attr}={val} on token {ordinal} in sentence {sent_id} is not printed in the output."
        )
    return output_text

def process_annotation(annotation_text, sent_id=None):
    lines = annotation_text.strip().split("\n")
    token_descriptions = []
    multiword_tokens = {}  # key = start index, value = (start, end)

    for line in lines:
        if line.startswith("#"):
            continue
        if re.match(r"^\d+-\d+", line):  # Match something like 16-17
            match = re.match(r"^(\d+)-(\d+)", line)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                multiword_tokens[start] = (start, end)
            continue  # skip the multiword token line itself
        fields = line.split("\t")
        if len(fields) < 8 or "-" in fields[0] or "." in fields[0]:
            continue
        index = int(fields[0])
        if index in multiword_tokens:
            start, end = multiword_tokens[index]
            start_ordinal = get_ordinal(start)
            end_ordinal = get_ordinal(end)
            token_descriptions.append(f"The {start_ordinal} and {end_ordinal} tokens make up a single surface word.")
        token_desc = describe_token(fields, sent_id)
        token_descriptions.append(token_desc)

    return "\n".join(token_descriptions)

def process_conllu_file(filepath, output_json="prompts.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    sentences = content.strip().split("\n\n")
    prompt_dict = {}

    for sentence in sentences:
        sent_id = None
        lines = sentence.split("\n")
        token_lines = []
        for line in lines:
            if line.startswith("#"):
                match = re.match(r"#\s*sent_id\s*=\s*(\S+)", line)
                if match:
                    sent_id = match.group(1)
            else:
                token_lines.append(line)
        if not token_lines:
            continue
        annotation_text = "\n".join(token_lines)
        prompt = process_annotation(annotation_text, sent_id)
        if not sent_id:
            sent_id = "unknown"
        prompt_dict[sent_id] = prompt

    with open(output_json, "w", encoding="utf-8") as out_file:
        json.dump(prompt_dict, out_file, indent=4, ensure_ascii=False, sort_keys=True)
    print(f"Prompts saved to {output_json}")

    with open("unprinted_features.log", "w", encoding="utf-8") as log_file:
        for warning in global_warnings:
            log_file.write(warning + "\n")
        log_file.write(f"Total warnings: {len(global_warnings)}\n")
    print("Unprinted feature warnings saved to unprinted_features.log")

    return prompt_dict

def generate_prompt_from_annotation(annotation_text, sent_id=None, log_warnings=False, print_warnings=True):
    """
    Generate a prompt directly from annotation text without saving to a file.

    Args:
        annotation_text (str): CoNLL-U format annotation text
        sent_id (str, optional): Sentence ID for warning logs
        log_warnings (bool): Whether to log warnings to a file
        print_warnings (bool): Whether to print warnings to console

    Returns:
        str: Generated prompt describing the annotated tokens
    """
    # Clear previous warnings
    global global_warnings
    global_warnings = []

    result = process_annotation(annotation_text, sent_id)

    # Log warnings if requested
    if log_warnings and global_warnings:
        log_dir = Path("warnings")
        log_dir.mkdir(exist_ok=True)
        log_filename = log_dir / f"warnings_{sent_id}.log"
        with open(log_filename, "w", encoding="utf-8") as log_file:
            for warning in global_warnings:
                log_file.write(warning + "\n")
            log_file.write(f"Total warnings: {len(global_warnings)}\n")

        # Only print if requested
        if print_warnings:
            print(f"Warnings for sentence {sent_id} saved to {log_filename}")

    # Print warnings to console only if requested
    if print_warnings and global_warnings:
        print(f"Sentence {sent_id} has {len(global_warnings)} warnings:")
        for warning in global_warnings[:5]:  # Show the first 5 warnings
            print(f"  - {warning}")
        if len(global_warnings) > 5:
            print(f"  ... and {len(global_warnings) - 5} more warnings")

    return result

if __name__ == "__main__":
    conllu_filename = "sentences_fixed.conllu"
    process_conllu_file(conllu_filename, "sentences_fixed_prompts.json")
