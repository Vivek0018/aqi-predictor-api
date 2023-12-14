fetch('https://fast-api-new.onrender.com/')
    .then((response) => response.json())
    .then((data) => console.log(data))