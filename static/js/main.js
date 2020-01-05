$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.image-result').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();

        // $('.image-result').show();

        // $('#imageResult').show(function() {
        //     // $(this).css('background-image', 'url("5-giay1.jpg")');
        //     $(this).attr('src', "{{ url_for('static', filename='result/' + 'res_0.jpg') }}");
        // });

        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            dataType: 'json',
            success: function (data) {
                // Get and display the result
                $('.loader').hide();

                $('.image-result').show();
                $('#imageResult').attr('src', '/static/result/res_' + data['image_name'])
                $('#imageResult').show();

                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
                console.log('Success!');
            },
        });
    });

});
