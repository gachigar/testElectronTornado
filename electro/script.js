function musSelect(mus) {
    document.getElementById('audioSourceUrl').src = mus;
    document.querySelector('audio').load();
    document.querySelector('audio').play();
}

function getMus() {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open("GET", "http://127.0.0.1:8000/", false);
    xmlHttp.send(null);
    return JSON.parse(xmlHttp.responseText);
}

function renderMus() {
    musArray = getMus();
    for (i of musArray) {
        let div = document.createElement('div');
        div.innerHTML = i;
        div.className = "musElement"
        div.setAttribute("onclick", 'musSelect("http://127.0.0.1:8000/files/' + i + '");');
        document.getElementById("musSelector").appendChild(div);
    }
}

renderMus();
