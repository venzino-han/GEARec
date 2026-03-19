
# Prompt template definitions
PROMPT_TEMPLATES = {
    'seq_rec': {
    'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Your task is to infer one or multiple kinds of products they may want to buy next, and generate relevant query terms that can be used to search for these potential products.\n' +\
                'Below is the user purchase history:\n',
    'tail': 'Based on this user\'s purchase history, generate relevant query terms that can be used to search for these potential products.',
    },
    
    'key_selection': {
        'head': 'You are an intelligent shopping assistant that helps predict what users may want to purchase next. Below is a list of items a user has purchased recently.\n' +\
                'Analyze the user\'s purchase history and infer their NEXT purchase intent driven by a specific past item.\n' + \
                'Below is the user purchase history:\n',
                
        'tail': 'First, identify the single key item from the history that most strongly motivates the user\'s next purchase.\n' + \
                'Then, generate a detailed, paragraph-length intent (3-5 sentences) describing the user\'s underlying preferences.\n' + \
                'This paragraph should clearly explain WHY the key item influenced them, WHAT specific attributes or features they value, and HOW this leads to their next purchase.\n\n' + \
                '# Output Format:\n' + \
                'Strictly follow this exact tag format:\n' + \
                '```\n<key_item>[Key Item Name]</key_item>\n<query>[A detailed paragraph describing the user\'s preferences and future intent]</query>\n```\n',
        },
}



PROMPT_TEMPLATES_YELP = {
    'seq_rec': {
        'head': 'You are an intelligent local guide that helps predict what businesses users may want to visit next. Below is a list of places a user has visited recently.\n' +
                'Your task is to infer one or multiple kinds of businesses (e.g., restaurants, services) they may want to visit next, and generate relevant query terms that can be used to search for these potential places.\n' +
                'Below is the user visit history:\n',
        'tail': 'Based on this user\'s visit history, generate relevant query terms that can be used to search for these potential businesses.',
    },

    'key_selection': {
        'head': 'You are an intelligent local guide that helps predict what businesses users may want to visit next. Below is a list of places a user has visited recently.\n' +\
                'Analyze the user\'s visit history and infer their NEXT visit intent driven by a specific past place.\n' + \
                'Below is the user visit history:\n',
                
        'tail': 'First, identify the single key place from the history that most strongly motivates the user\'s next visit.\n' + \
                'Then, generate a detailed, paragraph-length intent (3-5 sentences) describing the user\'s underlying preferences.\n' + \
                'This paragraph should clearly explain WHY the key place influenced them, WHAT specific attributes, services, or vibes they value, and HOW this leads to their next visit.\n\n' + \
                '# Output Format:\n' + \
                'Strictly follow this exact tag format:\n' + \
                '```\n<key_place>[Key Place Name]</key_place>\n<query>[A detailed paragraph describing the user\'s preferences and future visit intent]</query>\n```\n',
    },
}