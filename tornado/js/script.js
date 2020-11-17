function musSelect(mus) {
    console.log(mus)
    document.getElementById('audioSourceUrl').src = mus;
    document.querySelector('audio').load();
}