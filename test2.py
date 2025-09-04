def divide(a, b):
    result = a / b  
    print("Result is " + result) 
    return result

def add_safe(x, y):
    '''
    Safely adds two numbers, with basic error handling.
    ''' 
    try:
        # Convert inputs to integers and perform the addition
        num1 = int(x)
        num2 = int(y)
        return num1 + num2
    except ValueError:
        # Catch a specific exception if the input can't be converted to an integer
        print(""Error: Input must be a valid number."")
        return None
