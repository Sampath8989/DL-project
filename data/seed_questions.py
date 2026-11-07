"""Seed evaluation set: short-answer factual questions with gold strings.

Each entry is (question, [accepted answer substrings], tier).

Tiers are a *popularity* axis, not a reasoning-difficulty axis:
    easy    high-frequency facts almost certain to be in pretraining
    medium  mid-frequency facts
    hard    long-tail facts a sub-1B model is likely to confabulate

Tiering matters because the study needs hallucinations to actually occur
at small scale, and because tier is one of the confounds the evaluation
must control for (a probe that has merely learned "this looks like a hard
question" is not a truth probe).

DESIGN NOTE ON FORMAT BALANCE
    Difficulty is manipulated by the *frequency of the entity*, not by the
    shape of the question. Every question format (capital-of, authorship,
    year, superlative, unit, count, symbol) is meant to appear at all three
    tiers, so that tier cannot be decoded from surface form alone. Where a
    format is still tier-skewed, `format_audit()` below will show it.

Labelling rule: a generated answer is correct if any accepted substring
appears in it, case-insensitively; label 1 = hallucinated.

Provenance: merged from the two banks authored for Review 1 and Review 2,
deduplicated on the question string.
"""

SEED = [

    # ---------------- easy ----------------
    ("How many bones are in the adult human body?", ["206"], "easy"),
    ("How many continents are there on Earth?", ["seven", "7"], "easy"),
    ("How many minutes are there in a full day?", ["1440", "1,440"], "easy"),
    ("How many players from one team are on the pitch in football?",
     ["eleven", "11"], "easy"),
    ("How many sides does a hexagon have?", ["six", "6"], "easy"),
    ("How many strings does a standard guitar have?", ["six", "6"], "easy"),
    ("In which continent is the Sahara desert?", ["africa"], "easy"),
    ("In which country are the Pyramids of Giza?", ["egypt"], "easy"),
    ("What does DNA stand for?", ["deoxyribonucleic"], "easy"),
    ("What gas do plants absorb for photosynthesis?",
     ["carbon dioxide", "CO2"], "easy"),
    ("What gas do plants absorb from the air for photosynthesis?",
     ["carbon dioxide", "co2"], "easy"),
    ("What is the boiling point of water at sea level in Celsius?", ["100"], "easy"),
    ("What is the capital city of Australia?", ["canberra"], "easy"),
    ("What is the capital city of Brazil?", ["brasilia", "brasília"], "easy"),
    ("What is the capital city of Canada?", ["ottawa"], "easy"),
    ("What is the capital city of Egypt?", ["cairo"], "easy"),
    ("What is the capital city of France?", ["paris"], "easy"),
    ("What is the capital city of Italy?", ["rome"], "easy"),
    ("What is the capital city of Japan?", ["tokyo"], "easy"),
    ("What is the capital city of Kenya?", ["nairobi"], "easy"),
    ("What is the capital of Australia?", ["Canberra"], "easy"),
    ("What is the capital of Brazil?", ["Brasilia", "Brasília"], "easy"),
    ("What is the capital of Canada?", ["Ottawa"], "easy"),
    ("What is the capital of China?", ["Beijing", "Peking"], "easy"),
    ("What is the capital of Egypt?", ["Cairo"], "easy"),
    ("What is the capital of France?", ["Paris"], "easy"),
    ("What is the capital of Germany?", ["Berlin"], "easy"),
    ("What is the capital of India?", ["New Delhi", "Delhi"], "easy"),
    ("What is the capital of Italy?", ["Rome"], "easy"),
    ("What is the capital of Japan?", ["Tokyo"], "easy"),
    ("What is the capital of Mexico?", ["Mexico City"], "easy"),
    ("What is the capital of Russia?", ["Moscow"], "easy"),
    ("What is the capital of South Korea?", ["Seoul"], "easy"),
    ("What is the capital of Spain?", ["Madrid"], "easy"),
    ("What is the chemical formula for water?", ["H2O", "H₂O"], "easy"),
    ("What is the chemical symbol for gold?", ["au"], "easy"),
    ("What is the chemical symbol for iron?", ["Fe"], "easy"),
    ("What is the chemical symbol for oxygen?", ["O"], "easy"),
    ("What is the chemical symbol for sodium?", ["na"], "easy"),
    ("What is the currency of Japan?", ["yen"], "easy"),
    ("What is the freezing point of water in Celsius?", ["0", "zero"], "easy"),
    ("What is the freezing point of water in degrees Celsius?", ["0", "zero"], "easy"),
    ("What is the largest mammal on Earth?", ["blue whale", "whale"], "easy"),
    ("What is the largest mammal?", ["blue whale", "whale"], "easy"),
    ("What is the largest ocean on Earth?", ["pacific"], "easy"),
    ("What is the largest planet in the Solar System?", ["Jupiter"], "easy"),
    ("What is the longest river in Africa?", ["nile"], "easy"),
    ("What is the smallest prime number?", ["2", "two"], "easy"),
    ("What is the square root of 144?", ["12", "twelve"], "easy"),
    ("What is the tallest mountain above sea level?", ["everest"], "easy"),
    ("Which country gifted the Statue of Liberty to the United States?",
     ["France"], "easy"),
    ("Which country is the Great Wall located in?", ["china"], "easy"),
    ("Which is the largest continent by land area?", ["Asia"], "easy"),
    ("Which language has the most native speakers in the world?",
     ["chinese", "mandarin"], "easy"),
    ("Which organ in the human body pumps blood?", ["heart"], "easy"),
    ("Which organ pumps blood through the human body?", ["heart"], "easy"),
    ("Which planet has the most prominent ring system?", ["Saturn"], "easy"),
    ("Which planet is closest to the Sun?", ["mercury"], "easy"),
    ("Which planet is known as the Red Planet?", ["mars"], "easy"),
    ("Which vitamin does human skin produce in sunlight?", ["vitamin D", "D"], "easy"),
    ("Who composed the symphony containing the Ode to Joy?", ["Beethoven"], "easy"),
    ("Who painted the Mona Lisa?", ["leonardo", "da vinci"], "easy"),
    ("Who wrote On the Origin of Species?", ["Darwin"], "easy"),
    ("Who wrote Pride and Prejudice?", ["Austen"], "easy"),
    ("Who wrote Romeo and Juliet?", ["Shakespeare"], "easy"),
    ("Who wrote the novel 1984?", ["Orwell"], "easy"),
    ("Who wrote the play Hamlet?", ["Shakespeare"], "easy"),
    ("Who wrote the play Romeo and Juliet?", ["shakespeare"], "easy"),

    # ---------------- medium ----------------
    ("In what year did the Second World War end?", ["1945"], "medium"),
    ("In what year was the first iPhone released?", ["2007"], "medium"),
    ("In which year did India gain independence?", ["1947"], "medium"),
    ("In which year did the Berlin Wall fall?", ["1989"], "medium"),
    ("In which year did the Chernobyl disaster occur?", ["1986"], "medium"),
    ("In which year did the Second World War end?", ["1945"], "medium"),
    ("In which year did the Titanic sink?", ["1912"], "medium"),
    ("What does the acronym GPU stand for?", ["graphics processing unit"], "medium"),
    ("What is the capital of Argentina?", ["Buenos Aires"], "medium"),
    ("What is the capital of Austria?", ["Vienna", "Wien"], "medium"),
    ("What is the capital of Chile?", ["Santiago"], "medium"),
    ("What is the capital of Greece?", ["Athens"], "medium"),
    ("What is the capital of Kenya?", ["Nairobi"], "medium"),
    ("What is the capital of Morocco?", ["Rabat"], "medium"),
    ("What is the capital of Nigeria?", ["Abuja"], "medium"),
    ("What is the capital of Norway?", ["Oslo"], "medium"),
    ("What is the capital of Peru?", ["Lima"], "medium"),
    ("What is the capital of Poland?", ["Warsaw"], "medium"),
    ("What is the capital of Portugal?", ["Lisbon"], "medium"),
    ("What is the capital of Sweden?", ["Stockholm"], "medium"),
    ("What is the capital of Switzerland?", ["bern"], "medium"),
    ("What is the capital of Thailand?", ["Bangkok"], "medium"),
    ("What is the capital of the Indian state of Kerala?",
     ["thiruvananthapuram", "trivandrum"], "medium"),
    ("What is the capital of Turkey?", ["Ankara"], "medium"),
    ("What is the capital of Vietnam?", ["Hanoi"], "medium"),
    ("What is the chemical formula of table salt?", ["nacl"], "medium"),
    ("What is the chemical symbol for silver?", ["Ag"], "medium"),
    ("What is the hardest naturally occurring mineral?", ["diamond"], "medium"),
    ("What is the hardest naturally occurring substance?", ["diamond"], "medium"),
    ("What is the largest desert in the world by area?",
     ["antarctic", "antarctica"], "medium"),
    ("What is the largest hot desert in the world?", ["Sahara"], "medium"),
    ("What is the largest island in the world?", ["greenland"], "medium"),
    ("What is the powerhouse of the cell?", ["mitochondri"], "medium"),
    ("What is the scientific study of earthquakes called?", ["seismology"], "medium"),
    ("What is the SI unit of electric current?", ["ampere", "amp"], "medium"),
    ("What is the SI unit of electrical resistance?", ["ohm"], "medium"),
    ("What is the smallest prime number greater than 20?", ["23"], "medium"),
    ("Which acid is present in the human stomach?", ["hydrochloric", "hcl"], "medium"),
    ("Which blood group is known as the universal donor?",
     ["o negative", "o-"], "medium"),
    ("Which country hosted the 2016 Summer Olympics?", ["brazil"], "medium"),
    ("Which element has atomic number 6?", ["carbon"], "medium"),
    ("Which element has the atomic number 6?", ["carbon"], "medium"),
    ("Which gas makes up about 78 percent of Earth's atmosphere?",
     ["nitrogen"], "medium"),
    ("Which gas makes up about 78 percent of the Earth's atmosphere?",
     ["nitrogen"], "medium"),
    ("Which Greek mathematician is known for a theorem about right triangles?",
     ["Pythagoras"], "medium"),
    ("Which is the smallest country in the world by area?", ["Vatican"], "medium"),
    ("Which landlocked salt lake between Israel and Jordan is famous for its salinity?",
     ["Dead Sea"], "medium"),
    ("Which metal is liquid at room temperature?", ["mercury"], "medium"),
    ("Which organelle is described as the powerhouse of the cell?",
     ["mitochond"], "medium"),
    ("Which planet has the most moons discovered so far?", ["saturn"], "medium"),
    ("Which planet is famous for its Great Red Spot?", ["Jupiter"], "medium"),
    ("Which programming language was created by Guido van Rossum?",
     ["python"], "medium"),
    ("Which river flows through the city of Varanasi?", ["ganga", "ganges"], "medium"),
    ("Which scientist formulated the three laws of motion?", ["Newton"], "medium"),
    ("Which scientist proposed the three laws of motion?", ["newton"], "medium"),
    ("Which sea separates Europe from Africa?", ["mediterranean"], "medium"),
    ("Which vitamin is produced in the skin on exposure to sunlight?",
     ["vitamin D"], "medium"),
    ("Who developed the theory of general relativity?", ["einstein"], "medium"),
    ("Who directed the 1975 film Jaws?", ["Spielberg"], "medium"),
    ("Who discovered penicillin?", ["Fleming"], "medium"),
    ("Who founded the Maurya Empire?", ["chandragupta"], "medium"),
    ("Who is credited with inventing the telephone?", ["Bell"], "medium"),
    ("Who is credited with inventing the World Wide Web?",
     ["berners-lee", "berners lee"], "medium"),
    ("Who painted The Starry Night?", ["Van Gogh", "Gogh"], "medium"),
    ("Who was the first person to walk on the Moon?", ["armstrong"], "medium"),
    ("Who was the first woman to win a Nobel Prize?", ["curie"], "medium"),
    ("Who wrote Crime and Punishment?", ["Dostoevsky", "Dostoyevsky"], "medium"),
    ("Who wrote Don Quixote?", ["Cervantes"], "medium"),
    ("Who wrote The Divine Comedy?", ["Dante"], "medium"),
    ("Who wrote The Great Gatsby?", ["Fitzgerald"], "medium"),
    ("Who wrote the Indian national anthem?", ["tagore"], "medium"),
    ("Who wrote the novel Pride and Prejudice?", ["austen"], "medium"),
    ("Who wrote the play A Doll's House?", ["Ibsen"], "medium"),
    ("Who wrote War and Peace?", ["Tolstoy"], "medium"),

    # ---------------- hard ----------------
    ("Deficiency of which vitamin causes scurvy?", ["vitamin C"], "hard"),
    ("In which country is the city of Timbuktu?", ["Mali"], "hard"),
    ("In which year was the transistor first demonstrated at Bell Labs?",
     ["1947"], "hard"),
    ("Titan is the largest moon of which planet?", ["Saturn"], "hard"),
    ("What does the acronym RAID stand for in storage systems?",
     ["redundant array"], "hard"),
    ("What is the capital of Bhutan?", ["Thimphu"], "hard"),
    ("What is the capital of Cambodia?", ["Phnom Penh"], "hard"),
    ("What is the capital of Croatia?", ["Zagreb"], "hard"),
    ("What is the capital of Cuba?", ["Havana", "Habana"], "hard"),
    ("What is the capital of Ecuador?", ["Quito"], "hard"),
    ("What is the capital of Estonia?", ["Tallinn"], "hard"),
    ("What is the capital of Finland?", ["Helsinki"], "hard"),
    ("What is the capital of Ghana?", ["Accra"], "hard"),
    ("What is the capital of Iceland?", ["Reykjavik", "Reykjavík"], "hard"),
    ("What is the capital of Ireland?", ["Dublin"], "hard"),
    ("What is the capital of Jordan?", ["Amman"], "hard"),
    ("What is the capital of Kazakhstan?", ["astana", "nur-sultan"], "hard"),
    ("What is the capital of Latvia?", ["Riga"], "hard"),
    ("What is the capital of Lebanon?", ["Beirut"], "hard"),
    ("What is the capital of Mongolia?", ["ulaanbaatar", "ulan bator"], "hard"),
    ("What is the capital of Myanmar?", ["Naypyidaw", "Nay Pyi Taw"], "hard"),
    ("What is the capital of Nepal?", ["Kathmandu"], "hard"),
    ("What is the capital of Slovenia?", ["Ljubljana"], "hard"),
    ("What is the capital of the Indian state of Arunachal Pradesh?",
     ["itanagar"], "hard"),
    ("What is the capital of Uruguay?", ["Montevideo"], "hard"),
    ("What is the capital of Venezuela?", ["Caracas"], "hard"),
    ("What is the capital of Zimbabwe?", ["Harare"], "hard"),
    ("What is the chemical symbol for tungsten?", ["W", "tungsten"], "hard"),
    ("What is the constitutional capital of Bolivia?", ["Sucre", "La Paz"], "hard"),
    ("What is the currency of Poland?", ["zloty", "złoty"], "hard"),
    ("What is the hardest known synthetic material after diamond?",
     ["boron nitride", "wurtzite"], "hard"),
    ("What is the largest city of Sri Lanka?", ["Colombo"], "hard"),
    ("What is the largest lake in Africa by area?", ["Victoria"], "hard"),
    ("What is the largest lake in Africa by surface area?", ["victoria"], "hard"),
    ("What is the official capital of Tanzania?", ["Dodoma"], "hard"),
    ("What is the SI unit of luminous intensity?", ["candela"], "hard"),
    ("What is the SI unit of magnetic flux?", ["weber"], "hard"),
    ("What is the time complexity of merge sort in the worst case?",
     ["n log n", "nlogn"], "hard"),
    ("What is the value of the Avogadro constant to three significant figures?",
     ["6.02"], "hard"),
    ("Which algorithm is used to find the shortest path in a weighted graph with non-negative edges?",
     ["dijkstra"], "hard"),
    ("Which chemist is credited with the periodic table of 1869?",
     ["Mendeleev"], "hard"),
    ("Which dynasty built the Brihadeeswarar temple at Thanjavur?", ["chola"], "hard"),
    ("Which element has atomic number 42?", ["molybdenum"], "hard"),
    ("Which element has the chemical symbol W?", ["tungsten", "wolfram"], "hard"),
    ("Which element is most abundant in the Sun?", ["hydrogen"], "hard"),
    ("Which element is named after the Greek word for hidden?", ["krypton"], "hard"),
    ("Which Indian city is known as the Manchester of India?", ["ahmedabad"], "hard"),
    ("Which Indian mathematician worked with G. H. Hardy at Cambridge?",
     ["ramanujan"], "hard"),
    ("Which Indian state has the longest coastline?", ["gujarat"], "hard"),
    ("Which is the longest bone in the human body?", ["femur"], "hard"),
    ("Which is the smallest of Earth's oceans?", ["Arctic"], "hard"),
    ("Which layer of the atmosphere contains the ozone layer?",
     ["stratosphere"], "hard"),
    ("Which mathematician introduced the notation for the summation sign?",
     ["euler"], "hard"),
    ("Which mountain range is traditionally taken to separate Europe from Asia?",
     ["Ural"], "hard"),
    ("Which physicist discovered radioactivity in 1896?", ["Becquerel"], "hard"),
    ("Which physicist formulated the uncertainty principle?", ["Heisenberg"], "hard"),
    ("Which physicist is credited with the discovery of the electron?",
     ["thomson"], "hard"),
    ("Which planet rotates on its side relative to its orbit?", ["Uranus"], "hard"),
    ("Which river flows through Baghdad?", ["Tigris"], "hard"),
    ("Which river is the longest in South America?", ["amazon"], "hard"),
    ("Which sea lies between Italy and the Balkans?", ["Adriatic"], "hard"),
    ("Which sorting algorithm has the best average-case behaviour among comparison sorts and is used in most standard libraries?",
     ["quicksort", "quick sort", "introsort", "timsort"], "hard"),
    ("Which strait separates Spain from Morocco?", ["Gibraltar"], "hard"),
    ("Which theorem states that no three positive integers satisfy a^n + b^n = c^n for n greater than two?",
     ["fermat"], "hard"),
    ("Who developed the first widely used polio vaccine?", ["Salk"], "hard"),
    ("Who discovered the neutron?", ["chadwick"], "hard"),
    ("Who formulated the uncertainty principle in quantum mechanics?",
     ["heisenberg"], "hard"),
    ("Who introduced the concept of entropy in thermodynamics?", ["clausius"], "hard"),
    ("Who painted Guernica?", ["Picasso"], "hard"),
    ("Who proposed the concept of the Turing machine?", ["turing"], "hard"),
    ("Who was the first Prime Minister of India?", ["Nehru"], "hard"),
    ("Who won the Nobel Prize in Literature in 1957?", ["Camus"], "hard"),
    ("Who wrote Brave New World?", ["Huxley"], "hard"),
    ("Who wrote One Hundred Years of Solitude?", ["Marquez", "Márquez"], "hard"),
    ("Who wrote the novel Things Fall Apart?", ["Achebe"], "hard"),
    ("Who wrote the Sanskrit grammar treatise Ashtadhyayi?",
     ["panini", "pāṇini"], "hard"),
    ("Who wrote the treatise Arthashastra?", ["kautilya", "chanakya"], "hard"),
]

PROMPT_TEMPLATE = (
    "Answer the question in at most five words. "
    "If you do not know, still give your best single answer.\n"
    "Question: {q}\nAnswer:"
)


def as_items():
    """Return extraction-ready dicts."""
    return [dict(prompt=PROMPT_TEMPLATE.format(q=q), question=q,
                 gold=gold, task=tier, idx=i)
            for i, (q, gold, tier) in enumerate(SEED)]


def label(item, generated_text):
    """1 = hallucinated (no accepted gold substring present)."""
    t = generated_text.lower()
    return 0 if any(g.lower() in t for g in item["gold"]) else 1


FORMATS = [
    ("capital", r"\bcapital\b"),
    ("symbol/formula", r"chemical symbol|chemical formula"),
    ("authorship", r"^who (wrote|painted|composed|directed)"),
    ("year", r"\bin (which|what) year\b|\byear (did|was)\b"),
    ("person/discovery",
     r"^who (discovered|developed|founded|invented|proposed|formulated|"
     r"is credited|was the first|won)"),
    ("superlative", r"\b(largest|longest|smallest|tallest|hardest|most)\b"),
    ("unit", r"\bSI unit\b|\bunit of\b"),
    ("count", r"^how many\b"),
]


def question_format(q):
    import re
    ql = q.lower()
    for name, pat in FORMATS:
        if re.search(pat, ql):
            return name
    return "other"


def format_audit():
    """Cross-tabulate question format against tier.

    A format that appears at only one tier is a shortcut: the probe could
    read the format instead of the truth. Run this before trusting any
    result that uses the tier as a confound.
    """
    from collections import Counter, defaultdict
    tab = defaultdict(Counter)
    for q, _, t in SEED:
        tab[t][question_format(q)] += 1
    return {t: dict(c) for t, c in tab.items()}


if __name__ == "__main__":
    from collections import Counter
    print(len(SEED), Counter(t for _, _, t in SEED))
    tiers = ("easy", "medium", "hard")
    audit = format_audit()
    names = [n for n, _ in FORMATS] + ["other"]
    print(f"{'format':18s}" + "".join(f"{t:>8s}" for t in tiers) + "   total")
    for n in names:
        row = [audit[t].get(n, 0) for t in tiers]
        flag = "  <- tier-skewed" if min(row) == 0 and sum(row) >= 4 else ""
        print(f"{n:18s}" + "".join(f"{v:8d}" for v in row)
              + f"{sum(row):8d}" + flag)
