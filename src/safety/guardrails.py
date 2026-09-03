def is_safe_query(query: str) -> tuple[bool, str]:
    """
    Checks user input against basic safety and relevance rules 
    for the SmartHire career portal.
    """
    if not query or len(query.strip()) < 3:
        return False, "Query is too short. Please ask a valid career or resume-related question."
    
    forbidden_topics = ["hack", "exploit", "malware", "violence", "illegal"]
    query_lower = query.lower()
    
    for word in forbidden_topics:
        if word in query_lower:
            return False, "I cannot process this request due to safety policies."
            
    return True, "Safe"