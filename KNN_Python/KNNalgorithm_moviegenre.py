import math
from operator import itemgetter
import matplotlib.pyplot as plt


VALID_GENRES = ("Romance", "Action", "?")

class movie:
    def __init__(self,name,kicks,kisses,genre):
        if genre not in VALID_GENRES:
            raise ValueError("Invalid Genre!  ('"+genre+"')")
        else:
            self.moviedata = [name,kicks,kisses,genre]



class dataset():
    def __init__(self):
        self.movies = []

    def add_movie(self,movieobj):
        self.movies.append(movieobj.moviedata)


    def calc_square_root_distance(self,m2bc):
        distance = []

        for i in range(len(self.movies)):
            dict = {}
            dict['name'] = self.movies[i][0]
            dict['eucli'] = math.sqrt( math.pow((m2bc.moviedata[1] - self.movies[i][1]),2) + math.pow((m2bc.moviedata[2] - self.movies[i][2]),2) )
            dict['genre'] = self.movies[i][3]
            distance.append(dict)
        return distance


    def sorted_distances(self,ed):
        return sorted(ed, key=itemgetter('eucli'))






training_data = dataset()

# sample data    ["Movie","#Kicks","#Kisses","Genre"]
training_data.add_movie(movie("California Man", 3, 104, "Romance"))
training_data.add_movie(movie("He's not reall into dudes", 2, 100, "Romance"))
training_data.add_movie(movie("Beautiful Woman", 1, 81, "Romance"))
training_data.add_movie(movie("Kevin Longblade", 101, 10, "Action"))
training_data.add_movie(movie("Robo Slayer 3000", 99, 5, "Action"))
training_data.add_movie(movie("Amped", 98, 2, "Action"))
training_data.add_movie(movie("Logan", 70, 30, "Action"))
training_data.add_movie(movie("Black Panther ", 100, 40, "Action"))
training_data.add_movie(movie("Dunkirk", 60, 6, "Action"))
training_data.add_movie(movie("Inception", 80, 45, "Action"))
training_data.add_movie(movie("Star Wars: The Last Jedi", 89, 56, "Action"))
training_data.add_movie(movie("Star Wars: The Force Awakens", 90, 38, "Action"))
training_data.add_movie(movie("Captain America: The Winter Soldier", 96, 65, "Action"))
training_data.add_movie(movie("Avatar", 89, 66, "Action"))
training_data.add_movie(movie("Deadpool", 50, 50, "Action"))
training_data.add_movie(movie("Doctor Strange", 69, 24, "Action"))
training_data.add_movie(movie("Fury", 84, 21, "Action"))
training_data.add_movie(movie("The Shape of Water", 23,102, "Romance"))
training_data.add_movie(movie("Call Me by Your Name", 25, 99, "Romance"))
training_data.add_movie(movie("Love, Simon", 12, 103, "Romance"))
training_data.add_movie(movie("Titanic", 49, 100, "Romance"))
training_data.add_movie(movie("Gone with the Wind", 33, 77, "Romance"))
training_data.add_movie(movie("Adrift", 43, 67, "Romance"))
training_data.add_movie(movie("Taken", 109, 5, "Action"))
training_data.add_movie(movie("Wonder Woman", 87, 25, "Action"))
training_data.add_movie(movie("The Equalizer", 78, 66, "Action"))
training_data.add_movie(movie("Mad Max: Fury Road", 66, 66, "Action"))
training_data.add_movie(movie("Ready Player One", 62, 55, "Action"))
training_data.add_movie(movie("Avengers Assemble", 55, 33, "Action"))




# movie to be classified
m2bc = movie("Up", 18, 90, "?")

# create an empty list to store the distances
euclidean_distance = []

# calculate the distance from to unknown film to all the others
euclidean_distance =training_data.calc_square_root_distance(m2bc)

# print the sorted list
sorted_distances = training_data.sorted_distances(euclidean_distance)

# Define a k (the k from KNN). This k is, at how many film you want to look.
# k should be smaller than the sample size to kick out outliers.
# Bug big enough to get a good vote!
K = 15

# Count the genres and derive a vote
candidates = list(sorted_distances[:K])

# counting candidates to predict nearest genre
genre_count = {}


for elem in candidates:
    g = elem['genre']
    if g not in genre_count:
        genre_count[g] = 1
    else:
        genre_count[g] += 1



# max count
nearest_genre = max(genre_count,key=genre_count.get)

# Apply the voting results to the unknown film
print("Before Prediction  :  %s" %m2bc.moviedata)
m2bc.moviedata[3] = nearest_genre
print("After Prediction  :  %s" %m2bc.moviedata)






# matplotlib to scatter plot


# plot training data

# plotting_list =   [ ['Romance', [xcoord] , [ycoord] ] ,  ['Action' ,  [xcoord] , [ycoord] ]  ]
plotting_list = []

for i in range(len(training_data.movies)):
    inserted_coord = False
    for j in range(len(plotting_list)):
        if training_data.movies[i][3] == plotting_list[j][0]:
            plotting_list[j][1].append(training_data.movies[i][1])
            plotting_list[j][2].append(training_data.movies[i][2])
            inserted_coord = True
    if inserted_coord == False:
        plotting_list.append([  training_data.movies[i][3],[training_data.movies[i][1]],[training_data.movies[i][2]]  ])


for p in range(len(plotting_list)):
    plt.scatter(plotting_list[p][1] , plotting_list[p][2] , label=plotting_list[p][0])


# plot new data
plt.scatter(m2bc.moviedata[1],m2bc.moviedata[2],label="New_Data:"+m2bc.moviedata[3])


plt.xlabel("No of Kicks")
plt.ylabel("No of Kisses")                                                
plt.title("KNN Algorithm - Supervised learning to predict Movie Genre ")  
plt.legend()                                                              
                                                                          

plt.show()












