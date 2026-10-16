trait TestFamily: 'static {
    type Visitor<'a>: ?Sized + 'a;

    fn query_visitor<'call, 'out, V, A>(
        visitor: &'call mut QueryVisitOnce<'_, V, V::Output, A>,
    ) -> &'call mut Self::Visitor<'call>
    where
        V: QueryDistanceVisitor<'out>,
        A: AllocatorCore + std::fmt::Debug + Send + Sync + 'out;
}

trait VisitTestFamily {
    fn visit_l2(
        &mut self,
        kernel: Reified<L2, 1, AsData<1>, Scalar>,
    ) -> Result<(), quantizer::CompressionError>;
}

impl<'a, V, A> VisitTestFamily for QueryVisitOnce<'_, V, V::Output, A>
where
    V: QueryDistanceVisitor<'a>,
    A: AllocatorCore + std::fmt::Debug + Send + Sync + 'a,
{
    fn visit_l2(
        &mut self,
        kernel: Reified<L2, 1, AsData<1>, Scalar>,
    ) -> Result<(), quantizer::CompressionError> {
        let allocator = self.allocator.take().unwrap();
        let curried =
            kernel.prepare_and_curry(self.quantizer, self.query, allocator, self.scratch)?;

        self.visitor.visit(|v| v.visit(curried));
        Ok(())
    }
}

impl TestFamily for Bits<1> {
    type Visitor<'a> = dyn VisitTestFamily + 'a;

    fn query_visitor<'call, 'out, V, A>(
        visitor: &'call mut QueryVisitOnce<'_, V, V::Output, A>,
    ) -> &'call mut Self::Visitor<'call>
    where
        V: QueryDistanceVisitor<'out>,
        A: AllocatorCore + std::fmt::Debug + Send + Sync + 'out,
    {
        visitor
    }
}

trait TestDispatch {
    fn dispatch_distance<'a>(
        &self,
        quantizer: &SphericalQuantizer,
        visitor: &'a mut (dyn VisitTestFamily + 'a),
    );
}

impl TestDispatch for Bits<1> {
    fn dispatch_distance<'a>(
        &self,
        quantizer: &SphericalQuantizer,
        visitor: &'a mut (dyn VisitTestFamily + 'a),
    ) {
        visitor
            .visit_l2(Reified::new(
                quantizer.as_functor(),
                quantizer.output_dim(),
                Scalar,
            ))
            .unwrap()
    }
}
