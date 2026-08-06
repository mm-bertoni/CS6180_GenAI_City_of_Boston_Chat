TEST_CASES = [
    {
        "name": "Non-existant meeting",
        "prompt": "Tell me about the public meeting on beavers",
        "expectations": {
            "expected_agents": ["RAGAgent", "AnswerAgent"],
            "should_return_citation": False,
            "response_notes": "Should not find any matching public notices"
        },
    },
    {
        "name": "Non-existant meeting",
        "prompt": "Tell me about the city meeting on unicorns sightings",
        "expectations": {
            "expected_agents": ["RAGAgent", "AnswerAgent"],
            "should_return_citation": False,
            "response_notes": "Should not find any matching public notices"
        },
    },
    {
        "name": "Public testimony",
        "prompt": "Can I testify at the August 6th Tree Removal Hearing??"
        "expectations": {
            "expected_agents": ["RAGAgent", "AnswerAgent"],
            "should_return_citation": True,
            "response_notes": "Valid meeting but no public testimony at this hearing"
        },
    },
    {
        "name": "Public testimony",
        "prompt": "Can I testify at the August 6th Tree Removal Hearing??"
        "expectations": {
            "expected_agents": ["RAGAgent", "AnswerAgent"],
            "should_return_citation": True,
            "response_notes": "Valid meeting but no public testimony at this hearing"
        },
    },
    {
        "name": "Cancelled meeting",
        "prompt": "Is the Boston Landmarks Commission meeting on August 11th?"
        "expectations": {
            "expected_agents": ["RAGAgent", "AnswerAgent"],
            "should_return_citation": True,
            "response_notes": "Valid meeting but was cancelled"
        },
    },
    {
        "name": "Cancelled meeting",
        "prompt": "Is the Mission Hill Triangle Architectural hearing happening?"
        "expectations": {
            "expected_agents": ["RAGAgent", "AnswerAgent"],
            "should_return_citation": True,
            "response_notes": "Valid meeting but was cancelled"
        },
    },
    {
        "name": "Public Notice PDF details",
        "prompt": "What is Docket #0218 from the City Council Committee on Labor about?"
        "expectations": {
            "expected_agents": ["RAGAgent", "AnswerAgent"],
            "should_return_citation": True,
            "response_notes": "Docket PDF affiliated with NoticeID 16600016"
        },
    },
    {
        "name": "Public Notice PDF details",
        "prompt": "What is Docket #0932 from the City Council Committee on City Services about?"
        "expectations": {
            "expected_agents": ["RAGAgent", "AnswerAgent"],
            "should_return_citation": True,
            "response_notes": "Docket PDF affiliated with NoticeID 16600081"
        },
    },
    {
        "name": "Irrelevant Query",
        "prompt": "Explain crypto wallets"
        "expectations": {
            "expected_agents": ["IgnoreAgent", "AnswerAgent"],
            "should_return_citation": False,
            "response_notes": "Not relevant to city government"
        },
    },
    {
        "name": "Irrelevant Query",
        "prompt": "Write a fun limeric"
        "expectations": {
            "expected_agents": ["IgnoreAgent", "AnswerAgent"],
            "should_return_citation": False,
            "response_notes": "Not relevant to city government"
        },
    },
    {
        "name": "Archive notice not in DB",
        "prompt": "What happened at the City Council Meeting on October 18th, 2023?"
        "expectations": {
            "expected_agents": ["RAGAgent", "WebSearchAgent", "AnswerAgent"],
            "should_return_citation": True,
            "response_notes": "Other City Council Meetings are in DB, but this one is outside of the archive pull. Should get picked up by web search"
        },
    },
     {
        "name": "Archive notice not in DB",
        "prompt": "What happened at the South End Landmark District Commission meeting on April 2nd, 2024?"
        "expectations": {
            "expected_agents": ["RAGAgent", "WebSearchAgent", "AnswerAgent"],
            "should_return_citation": True,
            "response_notes": "This meeting is outside of the archive pull. Should get picked up by web search"
        },
    },
    {
        "name": "Archive notice",
        "prompt": "What happened at the July Air Pollution Control meeting?"
        "expectations": {
            "expected_agents": ["RAGAgent", "AnswerAgent"],
            "should_return_citation": True,
            "response_notes": "Meeting from July 15th should be in archive pull"
        },
    },
    {
        "name": "Archive notice",
        "prompt": "What City Council meetings happened in July?"
        "expectations": {
            "expected_agents": ["RAGAgent", "AnswerAgent"],
            "should_return_citation": True,
            "response_notes": "July council meeting should be in archive. May pull up other city council committee meetings"
        },
    },

    {
        "name": "Non-public notice government question",
        "prompt": "How do I apply for a business certificate through the city?"
        "expectations": {
            "expected_agents": ["WebSearchAgent", "AnswerAgent"],
            "should_return_citation": True,
            "response_notes": "Relevant to city government. Should get picked up by web search"
        },
    },
     {
        "name": "Non-public notice government question",
        "prompt": "Is there a city project happening at White Stadium? If so, tell me about it"
        "expectations": {
            "expected_agents": ["WebSearchAgent", "AnswerAgent"],
            "should_return_citation": True,
            "response_notes": "Relevant to city government. Should get picked up by web search from the project tracker"
        },
    },
]
