
# static_content.py

courses = [
    {
        "title": "English Basics",
        "description": """Welcome to the English Basics crash course! This course is designed to help beginners develop a strong foundation in the English language.""",
        "topics": [
            {
                "title": "Parts of Speech",
                "content": """Parts of speech are categories that describe the function of words in a sentence. There are 9 parts of speech:
1. **Nouns**: Words that name people, places, things, or ideas. Example: *dog*, *city*, *happiness*.
2. **Verbs**: Words that express actions or states. Example: *run*, *is*, *seem*.
3. **Adjectives**: Words that describe nouns. Example: *blue*, *fast*, *happy*.
4. **Adverbs**: Words that modify verbs, adjectives, or other adverbs. Example: *quickly*, *very*, *yesterday*.
5. **Pronouns**: Words that replace nouns. Example: *he*, *they*, *it*.
6. **Prepositions**: Words that show relationships between nouns and other words. Example: *on*, *under*, *between*.
7. **Conjunctions**: Words that connect clauses or sentences. Example: *and*, *but*, *or*.
8. **Interjections**: Words used to express emotion. Example: *Wow!*, *Ouch!*, *Hey!*.
9. **Articles**: Words (like *a*, *an*, *the*) used to define the specificity of nouns. Example: *a cat*, *the house*.""",
                "quiz": {
                    "question": "Which part of speech describes an action?",
                    "options": ["Noun", "Verb", "Adjective", "Adverb"],
                    "answer": "Verb"
                },
                "exercise": "Write a sentence using each part of speech."
            },
            {
                "title": "Basic Sentence Structure",
                "content": """A basic sentence in English follows the **subject-verb-object** order.
Example: *John (subject) reads (verb) a book (object).* Complex sentences may include clauses, phrases, and modifiers.

- **Compound Sentences**: Contain two or more independent clauses joined by a conjunction (e.g., *and*, *but*). Example: *I finished my work, and I went for a walk.*
- **Complex Sentences**: Contain an independent clause plus one or more dependent clauses. Example: *Although it was raining, we decided to go out.*""",
                "quiz": {
                    "question": "Identify the subject in the sentence: 'Mary loves chocolate.'",
                    "options": ["Mary", "loves", "chocolate", "loves chocolate"],
                    "answer": "Mary"
                },
                "exercise": "Write 5 sentences following the subject-verb-object structure."
            },
            {
                "title": "Tenses",
                "content": """Tenses indicate the time of action. The three main tenses are:
1. **Present Tense**: Describes actions happening now. Example: *She writes.*
2. **Past Tense**: Describes actions that happened in the past. Example: *He walked.*
3. **Future Tense**: Describes actions that will happen in the future. Example: *They will arrive.*

Each tense has four aspects: simple, continuous, perfect, and perfect continuous.
- **Perfect Continuous** (e.g., *has been writing*) shows an action that started in the past and continues up to the present or a specified time.""",
                "quiz": {
                    "question": "Which tense describes actions happening now?",
                    "options": ["Past", "Present", "Future", "Past Perfect"],
                    "answer": "Present"
                },
                "exercise": "Write sentences in present, past, and future tense."
            },
            {
                "title": "Common Punctuation Marks",
                "content": """Punctuation marks help clarify meaning in written language. Key punctuation marks include:
1. **Period (.)**: Used at the end of a sentence.
2. **Comma (,)**: Separates parts of a sentence or items in a list.
3. **Question Mark (?)**: Indicates a question.
4. **Exclamation Mark (!)**: Expresses strong emotion.
5. **Quotation Marks (\"\")**: Indicate direct speech or quotations.
6. **Apostrophe (')**: Shows possession or omission of letters.
7. **Colon (:)**: Introduces a list, quotation, or explanation.
8. **Semicolon (;)**: Connects independent clauses that are closely related in meaning.""",
                "quiz": {
                    "question": "Which punctuation mark indicates possession?",
                    "options": ["Period", "Comma", "Apostrophe", "Exclamation Mark"],
                    "answer": "Apostrophe"
                },
                "exercise": "Write a short paragraph using at least 5 different punctuation marks."
            },
            # ---------------- NEW TOPIC ADDED BELOW ----------------
            {
                "title": "Vocabulary Building",
                "content": """Building a strong vocabulary is essential for reading comprehension and effective communication.
Tips for improving vocabulary:
1. **Reading Widely**: Books, newspapers, and articles expose you to new words in context.
2. **Keeping a Vocabulary Journal**: Note down unfamiliar words, look up meanings, and revisit them regularly.
3. **Using Flashcards or Apps**: Tools like Anki or Quizlet can help you review words efficiently.
4. **Contextual Usage**: Practice new words in sentences to better retain their meaning.""",
                "quiz": {
                    "question": "Which habit can help you encounter new words naturally?",
                    "options": ["Sleeping", "Reading Widely", "Ignoring unfamiliar words", "Watching silent movies"],
                    "answer": "Reading Widely"
                },
                "exercise": "Record 10 new words you’ve encountered recently and use each in a sentence."
            }
        ]
    },
    {
        "title": "Computer Basics",
        "description": """Welcome to the Computer Basics crash course! This course introduces beginners to essential computer concepts and practical skills.""",
        "topics": [
            {
                "title": "What is a Computer?",
                "content": """A computer is an electronic device that processes data and performs tasks according to instructions (software). Key components include:
1. **CPU (Central Processing Unit)**: Executes instructions.
2. **RAM (Random Access Memory)**: Temporarily stores data for active processes.
3. **Storage**: Permanently stores data (HDD or SSD).
4. **Input Devices**: Devices like a keyboard or mouse that allow users to interact with the computer.
5. **Output Devices**: Devices like a monitor or printer that display results.
6. **GPU (Graphics Processing Unit)**: Handles rendering of images and video, especially for graphics-intensive tasks.""",
                "quiz": {
                    "question": "Which component executes instructions?",
                    "options": ["RAM", "Storage", "CPU", "Input Devices"],
                    "answer": "CPU"
                },
                "exercise": "List the main hardware components of a computer and describe their functions."
            },
            {
                "title": "Software vs Hardware",
                "content": """Software refers to programs and operating systems that run on a computer, while hardware refers to the physical components of the computer.
Examples:
- **Hardware**: CPU, RAM, hard drive, keyboard, monitor.
- **Software**: Windows, macOS, Microsoft Word, web browsers.

Additionally, there's **firmware** (a special kind of software embedded in hardware devices) and **device drivers** (software that helps the operating system interact with hardware).""",
                "quiz": {
                    "question": "Which of the following is an example of software?",
                    "options": ["CPU", "RAM", "Windows", "Hard Drive"],
                    "answer": "Windows"
                },
                "exercise": "Differentiate between hardware and software with examples."
            },
            {
                "title": "Operating Systems",
                "content": """An operating system (OS) manages computer hardware and software resources and provides common services for computer programs.
Popular operating systems include:
1. **Windows**: Developed by Microsoft.
2. **macOS**: Developed by Apple.
3. **Linux**: Open-source and available in various distributions.
4. **Android**: A mobile operating system developed by Google.
5. **iOS**: A mobile operating system developed by Apple for iPhones and iPads.""",
                "quiz": {
                    "question": "Which operating system is open-source?",
                    "options": ["Windows", "macOS", "Linux", "Android"],
                    "answer": "Linux"
                },
                "exercise": "Research and list 3 operating systems and their primary features."
            },
            {
                "title": "Basics of Excel",
                "content": """Microsoft Excel is a powerful spreadsheet application used for data analysis, visualization, and management.
Key Features:
1. **Cells and Ranges**: The intersection of rows and columns forms cells where data is entered.
2. **Formulas and Functions**: Perform calculations using formulas (e.g., =A1+B1) and built-in functions (e.g., =SUM(A1:A5)).
3. **Formatting**: Change the appearance of cells using font styles, colors, borders, and number formats.
4. **Charts**: Visualize data using bar charts, line charts, pie charts, etc.
5. **Pivot Tables**: Summarize large datasets quickly and interactively.
6. **Data Validation**: Restrict the type of data entered in a cell (e.g., only numbers, dates).
7. **Macros**: Automate repetitive tasks by recording a sequence of actions.""",
                "quiz": {
                    "question": "Which feature in Excel allows you to summarize large datasets interactively?",
                    "options": ["Charts", "Pivot Tables", "Formulas", "Data Validation"],
                    "answer": "Pivot Tables"
                },
                "exercise": "Create a simple Excel sheet with a sales dataset and generate a chart to visualize the data."
            },
            # ---------------- NEW TOPIC ADDED BELOW ----------------
            {
                "title": "Internet Basics",
                "content": """The internet is a global network of interconnected computers and servers. Key concepts include:
1. **Web Browsers**: Software for accessing websites (e.g., Chrome, Firefox).
2. **URLs (Uniform Resource Locators)**: The address of a webpage (e.g., https://www.example.com).
3. **Search Engines**: Tools like Google or Bing that help find information online.
4. **Email**: A method to send messages and files electronically.
5. **Online Safety**: Includes using strong passwords, avoiding suspicious links, and updating software regularly.""",
                "quiz": {
                    "question": "Which software is used to access websites on the internet?",
                    "options": ["Excel", "Web Browser", "Antivirus", "Notepad"],
                    "answer": "Web Browser"
                },
                "exercise": "Explain the difference between a web browser and a search engine, and list 3 popular examples of each."
            }
        ]
    },
    {
        "title": "Crypto Basics",
        "description": """Welcome to the Crypto Basics crash course! This course introduces you to the world of cryptocurrencies and blockchain technology.""",
        "topics": [
            {
                "title": "What is Cryptocurrency?",
                "content": """A cryptocurrency is a digital or virtual currency that uses cryptography for security. Unlike traditional currencies, cryptocurrencies operate on decentralized networks based on blockchain technology.
Examples of cryptocurrencies:
1. **Bitcoin (BTC)**: The first and most popular cryptocurrency.
2. **Ethereum (ETH)**: A platform for decentralized applications and smart contracts.
3. **Litecoin (LTC)**: A peer-to-peer cryptocurrency.
4. **Stablecoins** (e.g., USDT, USDC): Cryptocurrencies pegged to stable assets like the US dollar.
5. **Privacy Coins** (e.g., Monero): Offer enhanced anonymity.""",
                "quiz": {
                    "question": "Which of the following is the first cryptocurrency?",
                    "options": ["Ethereum", "Bitcoin", "Litecoin", "Ripple"],
                    "answer": "Bitcoin"
                },
                "exercise": "Research and list 5 cryptocurrencies and their use cases."
            },
            {
                "title": "Blockchain Technology",
                "content": """Blockchain is a distributed ledger technology that records transactions across many computers so that the record cannot be altered retroactively.
Key features:
1. **Decentralization**: No central authority controls the network.
2. **Immutability**: Once data is recorded, it cannot be changed.
3. **Transparency**: All transactions are visible to participants.
4. **Consensus Mechanisms** (e.g., Proof of Work, Proof of Stake): Determine how network participants agree on the validity of transactions.""",
                "quiz": {
                    "question": "Which feature of blockchain ensures that data cannot be altered once recorded?",
                    "options": ["Transparency", "Immutability", "Decentralization", "Consensus"],
                    "answer": "Immutability"
                },
                "exercise": "Explain the main features of blockchain technology with examples."
            },
            {
                "title": "How to Buy and Store Cryptocurrencies",
                "content": """To buy and store cryptocurrencies, follow these steps:
1. **Choose an Exchange**: Popular exchanges include Coinbase, Binance, and Kraken.
2. **Create an Account**: Register on the exchange and complete verification.
3. **Buy Cryptocurrency**: Use fiat currency to purchase cryptocurrencies.
4. **Store Securely**: Use a wallet to store your assets securely. Wallet types include:
   - **Hot Wallets**: Online wallets for frequent trading.
   - **Cold Wallets**: Offline wallets for long-term storage.
   - **Hardware Wallets** (e.g., Ledger, Trezor): Physical devices providing extra security.
5. **Consider Decentralized Exchanges (DEX)**: For trading directly from your wallet without a centralized intermediary.""",
                "quiz": {
                    "question": "Which type of wallet is more secure for long-term storage?",
                    "options": ["Hot Wallet", "Cold Wallet", "Exchange Wallet", "Mobile Wallet"],
                    "answer": "Cold Wallet"
                },
                "exercise": "Research and list 3 cryptocurrency exchanges and compare their features."
            },
            # ---------------- NEW TOPIC ADDED BELOW ----------------
            {
                "title": "NFTs & Digital Assets",
                "content": """Non-Fungible Tokens (NFTs) are unique digital assets that represent ownership or proof of authenticity of a specific item or piece of content, often using blockchain technology.
Key points:
1. **Uniqueness**: Each NFT has a unique identifier, making it different from other tokens.
2. **Use Cases**: Digital art, collectibles, in-game items, music, and more.
3. **Marketplaces**: Platforms like OpenSea or Rarible for buying and selling NFTs.
4. **Smart Contracts**: Govern the creation and transfer of NFTs, ensuring authenticity and transparent ownership records.""",
                "quiz": {
                    "question": "Which characteristic of NFTs ensures that each token is distinguishable from another?",
                    "options": ["Fungibility", "Uniqueness", "Centralization", "Identical Metadata"],
                    "answer": "Uniqueness"
                },
                "exercise": "Research one NFT project and summarize its purpose, benefits, and potential drawbacks."
            }
        ]
    }
]

