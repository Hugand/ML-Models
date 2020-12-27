paint = [[]]
isDragging = false

function setup() {
    createCanvas(400, 400);
    document.getElementById("clear-btn")
        .addEventListener("click", () => {
            paint = [[]]
        })

    document.getElementById("send-btn")
        .addEventListener("click", () => {
            var image = new Image();
            image.id = "pic";
            image.src = canvas.toDataURL();

            let csrfTkn = document.getElementsByName("csrfmiddlewaretoken")[0].value

            fetch(image.src)
            .then(res => res.blob())
            .then(blob => {
                const file = new File([blob], 'test.png', blob)

                let formData = new FormData()
    
                formData.append("csrfmiddlewaretoken", csrfTkn)
                formData.append("myfile", file)
    
                fetch("http://localhost:8000/upload/", {
                    method: "POST",
                    body: formData
                }).then(r => r.json())
                .then(r => {
                    document.getElementById("pred-val").innerHTML = "PREDICTED: "+r
                })
            })

        })
}

function draw() {
    isDragging = false
    background(0);
    stroke(255)
    strokeWeight(40)
    paint.forEach(strk => {
        strk.forEach((currPoint, i) => {
            if(i === 0) return
    
            prevPoint = strk[i-1]
            line(prevPoint.x, prevPoint.y, currPoint.x, currPoint.y)
            // ellipse(x, y, 10, 10)
        })
    })
}

function mouseDragged(event) {
    paint[paint.length-1].push({x: mouseX, y: mouseY})
    return false;
}

function mouseReleased() {
    paint.push([])
}

