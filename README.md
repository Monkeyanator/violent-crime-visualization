# monkeyanator.github.io
Simple data visualization for violent crimes in Louisville area. 

This project was an excellent learning experience- it enabled me to further famililarize myself with Python as it relates to data science topics and techniques. 

<a href="http://pandas.pydata.org/pandas-docs/stable/">Pandas</a> empowered me to load the CSV file into memory and perform complex operations on it- such as transforming the matrix, trimming columns, and initializing zero-hot columns. The <a href="http://docs.python-requests.org/en/master/">requests</a> library proved painless in querying the Google Maps API with addresses and retrieving their corresponding latitude-longitude pairs. 

<a href="http://scikit-learn.org/stable/">Scikit-learn</a>- the library I was most terrified of going into this project- felt simple to use, assuming you understood the underlying machine learning concepts (or even if you didn't, I doubt it would have been too difficult). I was able to train a decision tree based on the latitude-longitude pairs and the victim of the given crime. The visualization of this tree can be viewed in the dot.pdf file contained in the repo. 

For future work on this dataset, I hope to incorporate census data describing the racial demographic of various areas as well as average annual income. 

Uses data from the <a href="https://data.louisvilleky.gov/dataset/crime-data">Louisville Open Data site</a>. 