(function ($) {
    $('form').submit(function (event) {
        event.preventDefault();

        var data = new FormData(this);
        var action = function(d)
        {
            console.log(d);
        }
        $.ajax({
            url: '/upload/',
            data: data,
            type: "POST",
            contentType: false,
            processData: false,
            success: action,
            error: action,
        })
    })

}(jQuery))