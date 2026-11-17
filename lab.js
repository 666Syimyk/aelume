const pattern = document.querySelector('#pattern');
const rotation = document.querySelector('#rotation');
const brightness = document.querySelector('#brightness');
const mirror = document.querySelector('#mirror');
const target = document.querySelector('#target');

function updatePattern() {
  target.src = 'patterns/' + pattern.value + '_1080x1080.png';
  target.alt = pattern.selectedOptions[0].textContent;
}

function updateCalibration() {
  target.style.transform = 'rotate(' + rotation.value + 'deg) scaleX(' + (mirror.checked ? -1 : 1) + ')';
  target.style.filter = 'brightness(' + Number(brightness.value) / 100 + ')';
}

pattern.addEventListener('change', updatePattern);
rotation.addEventListener('change', updateCalibration);
brightness.addEventListener('input', updateCalibration);
mirror.addEventListener('change', updateCalibration);
