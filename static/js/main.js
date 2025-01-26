/*=============== SWIPER JS ===============*/
let swiperCards = new Swiper(".card__content", {
    loop: true,
    spaceBetween: 32,
    grabCursor: true,
  
    pagination: {
      el: ".swiper-pagination",
      clickable: true,
      dynamicBullets: true,
    },
  
    navigation: {
      nextEl: ".swiper-button-next",
      prevEl: ".swiper-button-prev",
    },
  
    breakpoints:{
      600: {
        slidesPerView: 2,
      },
      968: {
        slidesPerView: 3,
      },
    },
  });

  var leftArrow = document.getElementById("leftArrow");
  var rightArrow = document.getElementById("rightArrow");
  var slider = document.getElementById("Slider");
  
  let a = 0;
  rightArrow.style.visibility = "hidden";
  leftArrow.onclick = function() {
      a = a - 550;
      slider.style.left = a + "px";
      // right Arrow
      if(a == 0){
          rightArrow.style.visibility = "hidden";
      }else{
          rightArrow.style.visibility = "visible"
      }
      // Left Arrow
      if(a == -1650){
          leftArrow.style.visibility = "hidden";
      }else{
          leftArrow.style.visibility = "visible"
      }
  }
  rightArrow.onclick = function() {
      a = a + 550;
      slider.style.left = a + "px";
      // right Arrow
      if(a == 0){
          rightArrow.style.visibility = "hidden";
      }else{
          rightArrow.style.visibility = "visible"
      }
      // Left Arrow
      if(a == -1650){
          leftArrow.style.visibility = "hidden";
      }else{
          leftArrow.style.visibility = "visible"
      }
  }
  


  