fetch('https://fast-api-new.onrender.com/city/')
    .then((response) => response.json())
    .then((data) => console.log(data))