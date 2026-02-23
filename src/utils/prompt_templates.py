
# 프롬프트 템플릿 정의
PROMPT_TEMPLATES = {
    'seq_rec': {
    'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
    'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products.',
    },

    'seq_rec_anchor': {
    'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
    'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products.\n' +\
            'Response should be start with format: "This user recently purchased [HISTORY_ITEM_NAME], which represents this user\'s latest preferences for [Specific Attributes in History] by offering [Connecting Features]".'
    },

    # 'seq_rec': {
    # 'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
    #             'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
    #             'Below is the user purchase history:\n',
    # 'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products.\n' +\
    #         'The response should be start with "Considering the user\'s recent purchase of [Item Name], ..."',
    # },
    'seq_rec_new': {
    'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +
            'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +
            'Below is the user purchase history:\n',
    'tail': 'Based on this user\'s purchase history, generate multiple relevant query terms that can be used to search for these potential products.\n' +
            'Each query must follow the format: "[Item Name]: [Item Attributes, Related User Preferences]\n'
        #     'Don\'t include any other text in the response.',
    },
    'seq_rec_temp': {
    'head': (
        'You are an intelligent shopping assistant capable of analyzing user purchase patterns over time.\n'
        'Your goal is to predict the user\'s next immediate need based on their purchase history.\n'
        'The input history is provided with relative timestamps (e.g., "(D-0)" means today, "(D-5)" means 5 days ago).\n\n'
        'User Purchase History:\n'
    ),
    'tail': (
        '\nInstruction:\n'
        'Analyze the history and follow these 4 steps sequentially to generate the output:\n'
        '1. Reasoning (Thinking): First, analyze the user\'s recent purchase patterns and intent. Explicitly determine which time window is most relevant and why.\n'
        '2. Window Selection: Based on your reasoning, state the selected time window (e.g., "Last 7 Days").\n'
        '3. Item Filtering: List the item titles that strictly fall within that window to ensure grounding.\n'
        '4. Query Generation: Generate 3 diverse search queries based on the identified needs.\n\n'
        'Output Format (Strictly follow this XML structure):\n'
        '<thinking> [Your step-by-step reasoning about user intent and window selection] </thinking>\n'
        '<window> [Selected Window] </window>\n'
        '<items> [List of Item Titles in Window] </items>\n'
        '<query> [Generated Query 1], [Generated Query 2], [Generated Query 3] </query>\n'
    )
    },

    'seq_rec_temp_no_thinking': {
    'head': (
        'You are an intelligent shopping assistant capable of analyzing user purchase patterns over time.\n'
        'Your goal is to predict the user\'s next immediate need based on their purchase history.\n'
        'The input history is provided with relative timestamps (e.g., "(D-0)" means today, "(D-5)" means 5 days ago).\n\n'
        'User Purchase History:\n'
    ),
    'tail': (
        '\nInstruction:\n'
        'Analyze the history and follow these 4 steps sequentially to generate the output:\n'
        '1. Window Selection: Based on your reasoning, state the selected time window (e.g., "Last 7 Days").\n'
        '2. Item Filtering: List the item titles that strictly fall within that window to ensure grounding.\n'
        '3. Query Generation: Generate 3 diverse search queries based on the identified needs.\n\n'
        'Output Format (Strictly follow this structure):\n'
        '<window> [Selected Window] </window>\n'
        '<items> [List of Item Titles in Window] </items>\n'
        '<query> [Generated Query 1], [Generated Query 2], [Generated Query 3] </query>\n'
    )
    },

    'seq_rec_new_2': {
    'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +
            'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +
            'Below is the user purchase history:\n',
    'tail': 'Based on this user\'s purchase history, select one representative item from the history that highlights the user\'s taste.\n' +
            'Start the response strictly with the format: "[History Item Name]: [Related User Preference]", followed by a brief explanation connecting it to the recommendations.\n' +
            'Then, generate multiple relevant query terms for the potential next products.\n' +
            'Each query must strictly follow the format: "[Target Item Name]: [Item Attributes, Related User Preferences]".',
    },

    'seq_rec_date': {
    'head': '### Role\n' +\
            'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
            'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n\n' +\
            '### Task\n' +\
            'Determine the **"Current Interest Session Start Date"** relative to the Target Purchase Date.\n' +\
            'Analyze the user\'s history to find the point where the current shopping intent began.\n' +\
            'Consider both **Semantic Relevance** (is it related?) and **Temporal Proximity** (is it recent enough to matter?).\n' +\
            '\n### Input:\n',

    'tail': '### Instructions\n' +\
            '1. Calculate the time gap between items and the Target Date.\n' +\
            '2. Identify any significant "Time Breaks" (e.g., > 1 month of inactivity) that suggest a shift in intent.\n' +\
            '3. Determine the **Cut-off Date**. Items before this date are considered "Past History", and items after are "Current Context".\n' +\
            '4. Based on the user\'s purchase history, generate relevant query terms that can be used to search for these potential products.',
    },

    'seq_rec_recent': {
    'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
    'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products.\n' +\
            'The response should be start with "Considering the user\'s most recent purchase of [Last Item Name], ..."',
    },

    'seq_rec_recent2': {
    'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
    'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products.\n' +\
            'The response should be start with "Considering the user\'s recent purchase of [Item Name], ..."',
    },

    'seq_rec_with_recommendation': {
    'head': 'Analyze in depth and finally recommend next beauty product I might purchase inside <answer> and </answer>.\n' +\
            'For example, <answer>a product</answer>.\n\n' +\
            'Below is my historical product purchases:\n',
    'tail': "",
    },

    'item_summary': {
        'head': 'Summarize key attributes of the following beauty product inside <answer> and </answer>.\n\n',
        'tail': '',
    },

    'preference': {
        'head': '# User Purchase History',
        'tail': '# Task\nBased on this user\'s purchase history, describe user\'s preference:',
    },
    'next_item': {
        'head': '# User Purchase History',
        'tail': '# Task\nBased on this user\'s purchase history, predict what item the user will purchase next:',
    },
    'recommendation': {
        'head': '# User Purchase History',
        'tail': '# Task\nBased on this user\'s purchase history, recommend suitable items for the user:',
    },
    'user_profile': {
        'head': '# User Purchase History',
        'tail': '# Task\nBased on this user\'s purchase history, create a detailed user profile describing their interests and preferences:',
    },
    'recent_preference': {
        'head': '# User Purchase History',
        'tail': '# Task\nBased on this user\'s purchase history, describe user\'s most recent preference:',
    },
    'reasoning': {
        'head': '# User Purchase History',
        'tail': '# Task\nBased on this user\'s purchase history, reason about user\'s preference:',
    },
    'feature_reasoning_rec': {
        'head': 'You are an expert product analyst and shopping assistant. Below is a list of items a user has purchased recently.\n' +\
                 'Your task is to deeply analyze the underlying product features (such as brand, material, style, and functionality) that drive this user\'s choices.\n' +\
                 'Instead of looking at items in isolation, identify the common attributes that connect these purchases to infer the user\'s specific tastes.\n' +\
                 'Below is the user purchase history:\n',
        'tail': '# Task\n' +\
                'First, analyze and summarize the user\'s key preferences regarding specific product features (e.g., brands, materials, functionalities, or styles) derived from the history.\n' +\
                'Then, based on this feature summary, recommend suitable items that align with the user\'s taste:',
    },
    'feature_reasoning_recent_rec': {
        'head': 'You are an expert product analyst and shopping assistant. Below is a list of items a user has purchased recently.\n' +\
                 'Your task is to deeply analyze the underlying product features (such as brand, material, style, and functionality) that drive this user\'s choices.\n' +\
                 'Instead of looking at items in isolation, identify the common attributes that connect these purchases to infer the user\'s specific tastes, **giving more weight to recent interactions**.\n' +\
                 'Below is the user purchase history **(ordered by time)**:\n',
        'tail': '# Task\n' +\
                'First, analyze and summarize the user\'s key preferences regarding specific product features (e.g., brands, materials, functionalities, or styles) derived from the history, **focusing on their latest interests**.\n' +\
                'Then, based on this feature summary, recommend suitable items that align with the user\'s **current** taste:',
    },
    
    'seq_rec_with_sasrec': {
        'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
        'sasrec_section': '\n# SASRec Recommender Results\n' +\
                          'A collaborative filtering recommender (SASRec) has suggested the following items based on user behavior patterns. ' +\
                          '**Note: These recommendations are not always accurate and should be used as reference only** when generating your query terms:\n',
        'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products. ' +\
                'You may refer to the SASRec recommendations above, but they are not always correct, so use your own judgment based on the purchase history.',
    },
    
    'seq_rec_recent_with_sasrec': {
        'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
        'sasrec_section': '\n# SASRec Recommender Results\n' +\
                          'A collaborative filtering recommender (SASRec) has suggested the following items based on user behavior patterns. ' +\
                          '**Note: These recommendations are not always accurate and should be used as reference only** when generating your query terms:\n',
        'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products. ' +\
                'The response should start with "Considering the user\'s most recent purchase of [Last Item Name], ..."\n' +\
                'You may refer to the SASRec recommendations above, but they are not always correct, so use your own judgment based on the purchase history.',
    },

}



PROMPT_TEMPLATES_YELP = {
    # 1. Basic Sequence Recommendation for Yelp
    'seq_rec': {
        'head': 'You are an intelligent local guide that helps predict what businesses users may want to visit next. Below is a list of places a user has visited recently.\n' +
                'Your task is to infer one or multiple kinds of businesses (e.g., restaurants, services) they may want to visit next, and generate relevant query terms that can be used to search for these potential places.\n' +
                'Below is the user visit history:\n',
        'tail': 'Based on this user\'s visit history, generate relevant query terms that can be used to search for these potential businesses.',
    },

    # 2. "Considering..." start format
    'seq_rec_new': {
        'head': 'You are an intelligent local guide that helps predict what businesses users may want to visit next. Below is a list of places a user has visited recently.\n' +
                'Your task is to infer one or multiple kinds of businesses they may want to visit next, and generate relevant query terms that can be used to search for these potential places.\n' +
                'Below is the user visit history:\n',
        'tail': 'Based on this user\'s visit history, generate multiple relevant query terms that can be used to search for these potential businesses.\n' +
                'Each query must strictly follow the format: "[Business Category]: [Business Attributes, Related User Preferences]".\n' +
                'The response should start with "Considering the user\'s recent visit to [Business Name], ..."',
    },

    'seq_rec_anchor': {
        'head': 'You are an intelligent local guide that helps predict what businesses users may want to visit next. Below is a list of places a user has visited recently.\n' +
                'Your task is to infer one or multiple kinds of businesses they may want to visit next, and generate relevant query terms that can be used to search for these potential places.\n' +
                'Below is the user visit history:\n',
        'tail': 'Based on this user\'s visit history, generate relevant query terms that can be used to search for these potential businesses.\n' +
                'The response should be start with format: "This user recently visited [Visited Business Name], which represents this user\'s latest preferences for [Specific Attributes in History] by offering [Connecting Features]".',
    },

    # 3. "[History Item] -> Explanation -> Queries" format (Most Structured)
    'seq_rec_new_2': {
        'head': 'You are an intelligent local guide that helps predict what businesses users may want to visit next. Below is a list of places a user has visited recently.\n' +
                'Your task is to infer one or multiple kinds of businesses they may want to visit next, and generate relevant query terms that can be used to search for these potential places.\n' +
                'Below is the user visit history:\n',
        'tail': 'Based on this user\'s visit history, select one representative place from the history that highlights the user\'s taste.\n' +
                'Start the response strictly with the format: "[Visited Business Name]: [Related User Preference]", followed by a brief explanation connecting it to the recommendations.\n' +
                'Then, generate multiple relevant query terms for the potential next businesses.\n' +
                'Each query must strictly follow the format: "[Target Business Category]: [Business Attributes, Related User Preferences]".',
    },

        'seq_rec_recent2': {
    'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
    'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products.\n' +\
            'The response should be start with "Considering the user\'s recent purchase of [Item Name], ..."',
    },

    'seq_rec_recent2': {
        'head': 'You are an intelligent local guide that helps predict what businesses users may want to visit next. Below is a list of places a user has visited recently.\n' +
                'Your task is to infer one or multiple kinds of businesses they may want to visit next, and generate relevant query terms that can be used to search for these potential places.\n' +
                'Below is the user visit history:\n',
        'tail': 'Based on this user\'s visit history, generate multiple relevant query terms that can be used to search for these potential businesses.\n' +
                'The response should start with "Considering the user\'s recent visit to [Business Name], ..."',
    },
}