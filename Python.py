#The total number of semesters at school
first_year =2
total_no_of_years =4
total_number_of_semesters_at_school = first_year*total_no_of_years
print("total_number_of_semesters_at_school\n",total_number_of_semesters_at_school)
program = """Hello world\n"""
print(program)
days = ('Monday\n', 'Tuesday\n', 'Wednesday\n', 'Thursday\n', 'Friday\n', 'Saturday\n', 'Sunday\n')
for day in days:
    print(day)
    #Getting Type of a Variable
    age = 19
    year_of_study =1.2
    z = 'male'
    statement = True
    k = 5+6j
    days = ('Monday\n', 'Tuesday\n', 'Wednesday\n', 'Thursday\n', 'Friday\n', 'Saturday\n', 'Sunday\n')
    list = [ 'abcd', 786 , 2.23, 'john', 70.2 ]
    capiatl_City = {"USA": "Washington DC", "India":"New" "Delhi", "UK":"London", "KENYA":"Nairobi"}
    print(type(capiatl_City))
    print(type(age))
    print(type(year_of_study))
    print(type(z))
    print(type(statement))
    print(type(k))
    print(type(days))
    print(type(list))
    #Type Casting
a = str(4)
b = int(4)
c = float(4)
print("a=",a)
print("b=",b)
print("c =",c)
#Area and perimeter of a circle
radius = 7
pi = 3.142
area_of_circle = pi*radius**2
perimeter_of_circle = 2*pi*radius
print("Area=",area_of_circle)
print("Perimeter=",perimeter_of_circle)
#local and variables
x = 5
y = 10
def division():
   division = x / y
   return division
print(division())
#concatenation
str = 'Hello Gamer'
int = '911'
print(str)
print(str[0])
print(str[2:7])
print(str[3:])
print(('Hello Gamer\n') * 5)
print(str + " " + int)
#Python list
list = [ 'abcd', 786 , 2.23, 'john', 70.2 ]
tinylist = [123, 'john']
print (list)            # Prints complete list
print (list[0])         # Prints first element of the list
print (list[1:3])       # Prints elements starting from 2nd till 3rd 
print (list[2:])        # Prints elements starting from 3rd element
print (tinylist * 2)    # Prints list two times
print (list + tinylist) # Prints concatenated lists
#Python arithmetic operations
a = 9
b = 4
c = 0
c = a + b
print("a: {} b: {} a+b {}".format(a, b, c))
print("a: %d b: %d a+b: %d" % (a, b, c))
c = a - b
print("a: %d b: %d a-b: %d" %(a, b, c))
c = a * b
print("a: %d b: %d a*b: %d" %(a, b, c))
print("a: {} b: {} a*b: {}" .format(a, b, c))
c = a ** b
print("a: %d b: %d a**b: %d" %(a, b, c))
#python comparison operators
a = 10
b = 20
if ( a == b ):
    print ("a is equal to b")
else:
    print ("a is nowhere close to b")
if ( a < b ):
    print ("a is less than b")
else:
    print ("a is greater than b")
    if (a != b):
        print("a is not equal to b")
    else:
        print("a is equal to b")
        #Python Bitwise Operators
a = 60            # 60 = 0011 1100
b = 13            # 13 = 0000 1101
c = 0
c = a & b;        # 12 = 0000 1100
print ("Line 1 - Value of c is ", c)
c = a | b;        # 61 = 0011 1101
#Python user input
name = "ELVIS KIPLANGAT"
age = 19
course = "software engineering"
print("My name is" , name , "I am" , age , "years old" , "I am taking" , course, sep= '.')
print("My name is" , name , "I am" , age , "years old" , "I am taking" , course)
# python control flow
marks = 80
result = ""
if marks < 31:
   result = "Failed"
elif marks > 75:
   result = "Passed with distinction"
else:
   result = "Passed"
print(result)
#Python for loops
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
for i in months:
    print(i)
    #Python while loops
count = 1
while (count <= 9):
    print('The count is:', count)
    count = count + 1
    #python jump statements/break statements
y = 0
while y < 10:
    print(y)
    y = y + 1
    if y == 5:
        print("I am tired of counting")
        break
    #Python continue statement
for x in range(10):
    if x == 5:
        continue
    print(x)
    #Python pass statement
for letter in 'Python':
    if letter == 'h':
        pass
        print('This is pass block')
    print('Current Letter :', letter)
    #Python range() function
for x in range(6):
    print(x)
for x in range(3, 6):
    print(x)
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    days = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
for x in months:
    for y in days:
        print(x, y)        
