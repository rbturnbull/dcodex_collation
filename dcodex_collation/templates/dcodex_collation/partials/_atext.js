$(".atext-notes").change(function(){ //triggers change in all input fields including text type
    unsaved = true;
});

$(".atext-notes-save").click(function(e){
    var data = {
        'column': '{{ column.id }}', 
        'notes': $("#atext-notes-text").val(),
    }
    $.ajax({
        type: "POST",
        url: "{% url 'save_atext_notes' %}",
        data: data,
        success: function() {   
            unsaved = false;
            $("#atext-notes").text(data['notes']);
        }
    });
});
$(".atext").click(function(e){
    btn = $(this);
    var data = {
        'column': '{{ column.id }}', 
        'state': $(this).data('state'), 
    }
    if ($(this).hasClass("btn-danger")) {
        $.ajax({
            type: "POST",
            url: "{% url 'remove_atext' %}",
            data: data,
            success: function() {   
                $(".atext").removeClass("btn-danger");
                $(".atext").addClass("btn-outline-danger");
            }
        });
    }
    else {
        $.ajax({
            type: "POST",
            url: "{% url 'set_atext' %}",
            data: data,
            success: function() {  
                $(".atext").removeClass("btn-danger");
                $(".atext").addClass("btn-outline-danger");
                btn.addClass("btn-danger");
                btn.removeClass("btn-outline-danger");
            }
        });
    }
});