fetch('https://fast-api-demo.vercel.app/')
    .then((response) => response.json())
    .then((data) => console.log(data))