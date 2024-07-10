




def get_matching_bracket(sequence, start_index, open_type="(", close_type=")"):
    num_brackets = 1
    i = start_index
    
    # Iterate until we find the matching closing bracket
    while num_brackets >= 1:
        # If we find an opening bracket, increment the number of brackets
        if sequence[i] == open_type:
            num_brackets += 1
        # If we find a closing bracket, decrement the number of brackets
        elif sequence[i] == close_type:
            num_brackets -= 1
            
        # Iterate to the next part of the sequence
        i += 1
        
    return i-1